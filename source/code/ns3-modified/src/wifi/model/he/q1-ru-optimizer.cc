/*
 * SPDX-License-Identifier: GPL-2.0-only
 */

#include "q1-ru-optimizer.h"

#include "he-phy.h"

#include "ns3/mpdu-aggregator.h"
#include "ns3/wifi-phy.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <limits>
#include <numeric>
#include <set>

namespace ns3
{

namespace
{

struct ServiceOption
{
    std::size_t packetCount{0};
    uint32_t payloadBytes{0};
    uint32_t psduBytes{0};
    double utilityBytes{0.0};
    Time duration{0};
};

struct Layout
{
    uint8_t code;
    std::vector<HeRu::RuSpec> rus;
};

uint8_t
GetRuClassIndex(RuType type)
{
    switch (type)
    {
    case RuType::RU_26_TONE:
        return 1;
    case RuType::RU_52_TONE:
        return 2;
    case RuType::RU_106_TONE:
        return 3;
    case RuType::RU_242_TONE:
        return 4;
    default:
        return 0;
    }
}

const std::vector<Layout>&
GetLayouts()
{
    static const std::vector<Layout> layouts = [] {
        std::vector<Layout> result;
        std::set<std::array<std::size_t, 5>> compositions;
        for (const auto& [code, rus] : HeRu::m_heRuAllocations)
        {
            if (code > 192 || rus.empty())
            {
                continue;
            }
            if (!std::all_of(rus.begin(), rus.end(), [](const auto& ru) {
                    return ru.GetRuType() <= RuType::RU_242_TONE;
                }))
            {
                continue;
            }
            std::array<std::size_t, 5> composition{};
            for (const auto& ru : rus)
            {
                ++composition[GetRuClassIndex(ru.GetRuType())];
            }
            // Physical translations with the same RU-type multiset are objective-equivalent.
            if (compositions.insert(composition).second)
            {
                result.push_back({code, rus});
            }
        }
        return result;
    }();
    return layouts;
}

std::vector<ServiceOption>
GetServiceOptions(const Q1RuStationState& state,
                  RuType ruType,
                  const WifiTxVector& txVectorTemplate,
                  WifiPhyBand band,
                  Time maxDuration,
                  double delayWeight,
                  Time delayScale)
{
    std::vector<ServiceOption> options;
    if (state.queue.empty())
    {
        return options;
    }

    WifiTxVector txVector = txVectorTemplate;
    txVector.GetHeMuUserInfoMap().clear();
    txVector.SetHeMuUserInfo(state.aid,
                             {HeRu::RuSpec{ruType, 1, true}, state.mcs, state.nss});

    uint32_t payloadBytes{0};
    uint32_t ampduBytes{0};
    double utilityBytes{0.0};
    const auto scaleMs = std::max(1e-9, delayScale.ToDouble(Time::MS));
    for (std::size_t index = 0; index < state.queue.size(); ++index)
    {
        const auto& packet = state.queue[index];
        payloadBytes += packet.payloadBytes;
        utilityBytes += packet.payloadBytes *
                        (1.0 + delayWeight * std::max(0.0, packet.ageMs) / scaleMs);
        ampduBytes = MpduAggregator::GetSizeIfAggregated(packet.mpduBytes, ampduBytes);
        const auto psduBytes = index == 0 ? packet.mpduBytes : ampduBytes;
        if (index > 0 && (state.maxAmpduBytes == 0 || ampduBytes > state.maxAmpduBytes))
        {
            break;
        }
        const auto duration =
            WifiPhy::GetPayloadDuration(psduBytes, txVector, band, NORMAL_MPDU, state.aid);
        if (duration > maxDuration)
        {
            break;
        }
        options.push_back({index + 1, payloadBytes, psduBytes, utilityBytes, duration});
    }
    return options;
}

Time
GetLayoutHeader(const Layout& layout, const WifiTxVector& txVectorTemplate, uint8_t sigBMcs)
{
    WifiTxVector txVector = txVectorTemplate;
    txVector.GetHeMuUserInfoMap().clear();
    for (std::size_t slot = 0; slot < layout.rus.size(); ++slot)
    {
        txVector.SetHeMuUserInfo(static_cast<uint16_t>(slot + 1),
                                {layout.rus[slot], sigBMcs, 1});
    }
    txVector.SetRuAllocation({layout.code}, 0);
    return WifiPhy::CalculatePhyPreambleAndHeaderDuration(txVector);
}

const ServiceOption*
GetBestOption(const std::vector<ServiceOption>& options, Time threshold)
{
    const ServiceOption* best{nullptr};
    for (const auto& option : options)
    {
        if (option.duration <= threshold &&
            (!best || option.utilityBytes > best->utilityBytes))
        {
            best = &option;
        }
    }
    return best;
}

} // namespace

uint8_t
Q1RuOptimizer::GetRuClass(RuType type)
{
    return GetRuClassIndex(type);
}

Q1RuAction
Q1RuOptimizer::Optimize(const std::vector<Q1RuStationState>& states,
                        const WifiTxVector& txVector,
                        WifiPhyBand band,
                        std::size_t maxStations,
                        Time maxDuration,
                        Time exchangeOverhead,
                        double delayWeight,
                        Time delayScale)
{
    const auto count = std::min(states.size(), N_STATIONS);
    Q1RuAction bestAction;
    if (std::none_of(states.begin(), states.begin() + count, [](const auto& s) {
            return !s.queue.empty();
        }))
    {
        return bestAction;
    }

    using OptionTable = std::vector<std::vector<std::vector<ServiceOption>>>;
    OptionTable options(count, std::vector<std::vector<ServiceOption>>(N_RU_CLASSES));
    for (std::size_t station = 0; station < count; ++station)
    {
        for (uint8_t ruClass = 1; ruClass < N_RU_CLASSES; ++ruClass)
        {
            static const RuType types[] = {RuType::RU_TYPE_MAX,
                                           RuType::RU_26_TONE,
                                           RuType::RU_52_TONE,
                                           RuType::RU_106_TONE,
                                           RuType::RU_242_TONE};
            options[station][ruClass] = GetServiceOptions(states[station],
                                                          types[ruClass],
                                                          txVector,
                                                          band,
                                                          maxDuration,
                                                          delayWeight,
                                                          delayScale);
        }
    }

    const auto& layouts = GetLayouts();
    constexpr double NEGATIVE_INFINITY = -std::numeric_limits<double>::infinity();
    std::vector<uint8_t> sigBMcsValues;
    for (std::size_t station = 0; station < count; ++station)
    {
        if (!states[station].queue.empty())
        {
            sigBMcsValues.push_back(std::min<uint8_t>(5, states[station].mcs));
        }
    }
    std::sort(sigBMcsValues.begin(), sigBMcsValues.end());
    sigBMcsValues.erase(std::unique(sigBMcsValues.begin(), sigBMcsValues.end()),
                        sigBMcsValues.end());

    for (const auto& layout : layouts)
    {
        for (const auto sigBMcs : sigBMcsValues)
        {
            const auto headerDuration = GetLayoutHeader(layout, txVector, sigBMcs);
            if (headerDuration >= maxDuration)
            {
                continue;
            }
            const auto maxPayloadDuration = maxDuration - headerDuration;
            std::vector<Time> thresholds;
            for (const auto& ru : layout.rus)
            {
                const auto ruClass = GetRuClass(ru.GetRuType());
                for (std::size_t station = 0; station < count; ++station)
                {
                    if (std::min<uint8_t>(5, states[station].mcs) < sigBMcs)
                    {
                        continue;
                    }
                    const auto* option =
                        GetBestOption(options[station][ruClass], maxPayloadDuration);
                    if (option)
                    {
                        thresholds.push_back(option->duration);
                    }
                }
            }
            std::sort(thresholds.begin(), thresholds.end());
            thresholds.erase(std::unique(thresholds.begin(), thresholds.end()), thresholds.end());

            const auto slots = layout.rus.size();
            const auto maskCount = std::size_t{1} << count;
            for (const auto& threshold : thresholds)
            {
                std::vector<double> dp(maskCount, NEGATIVE_INFINITY);
                std::vector<std::vector<int>> parents(slots + 1,
                                                      std::vector<int>(maskCount, -1));
                dp[0] = 0.0;
                for (std::size_t slot = 0; slot < slots; ++slot)
                {
                    // A legal partition does not require every RU to be assigned.
                    std::vector<double> next = dp;
                    std::fill(parents[slot + 1].begin(), parents[slot + 1].end(), -2);
                    const auto ruClass = GetRuClass(layout.rus[slot].GetRuType());
                    for (std::size_t mask = 0; mask < maskCount; ++mask)
                    {
                        if (!std::isfinite(dp[mask]))
                        {
                            continue;
                        }
                        for (std::size_t station = 0; station < count; ++station)
                        {
                            if (static_cast<std::size_t>(std::popcount(mask)) >= maxStations ||
                                std::min<uint8_t>(5, states[station].mcs) < sigBMcs ||
                                (mask & (std::size_t{1} << station)) != 0)
                            {
                                continue;
                            }
                            const auto* option = GetBestOption(options[station][ruClass],
                                                               maxPayloadDuration);
                            if (!option || option->duration > threshold)
                            {
                                continue;
                            }
                            const auto nextMask = mask | (std::size_t{1} << station);
                            const auto value = dp[mask] + option->utilityBytes;
                            if (value > next[nextMask])
                            {
                                next[nextMask] = value;
                                parents[slot + 1][nextMask] = static_cast<int>(station);
                            }
                        }
                    }
                    dp = std::move(next);
                }

                for (std::size_t mask = 1; mask < maskCount; ++mask)
                {
                    if (static_cast<std::size_t>(std::popcount(mask)) > maxStations ||
                        !std::isfinite(dp[mask]))
                    {
                        continue;
                    }
                    bool hasMinimumMcs{false};
                    for (std::size_t station = 0; station < count; ++station)
                    {
                        hasMinimumMcs =
                            hasMinimumMcs ||
                            ((mask & (std::size_t{1} << station)) != 0 &&
                             std::min<uint8_t>(5, states[station].mcs) == sigBMcs);
                    }
                    if (!hasMinimumMcs)
                    {
                        continue;
                    }

                    std::vector<int> selected(slots, -1);
                    auto currentMask = mask;
                    bool valid = true;
                    for (std::size_t slot = slots; slot > 0; --slot)
                    {
                        const auto station = parents[slot][currentMask];
                        if (station == -1)
                        {
                            valid = false;
                            break;
                        }
                        if (station >= 0)
                        {
                            selected[slot - 1] = station;
                            currentMask &= ~(std::size_t{1} << station);
                        }
                    }
                    if (!valid)
                    {
                        continue;
                    }

                    Time payloadDuration{0};
                    std::vector<Q1RuAssignment> assignments;
                    for (std::size_t slot = 0; slot < slots; ++slot)
                    {
                        if (selected[slot] < 0)
                        {
                            continue;
                        }
                        const auto station = static_cast<std::size_t>(selected[slot]);
                        const auto ruClass = GetRuClass(layout.rus[slot].GetRuType());
                        const auto* option = GetBestOption(options[station][ruClass],
                                                           maxPayloadDuration);
                        payloadDuration = std::max(payloadDuration, option->duration);
                        assignments.push_back({station,
                                               layout.rus[slot],
                                               option->packetCount,
                                               option->payloadBytes,
                                               headerDuration + option->duration});
                    }
                    const auto ppduDuration = headerDuration + payloadDuration;
                    const auto denominator = (exchangeOverhead + ppduDuration).ToDouble(Time::S);
                    const auto objective = denominator > 0.0 ? dp[mask] * 8.0 / denominator : 0.0;
                    if (objective > bestAction.objective ||
                        (objective == bestAction.objective &&
                         ppduDuration < bestAction.ppduDuration))
                    {
                        bestAction =
                            {layout.code, std::move(assignments), objective, ppduDuration};
                    }
                }
            }
        }
    }
    return bestAction;
}

Q1RuAction
Q1RuOptimizer::Decode(const std::vector<Q1RuStationState>& states,
                      const std::vector<float>& ruLogits,
                      std::size_t maxStations)
{
    const auto count = std::min(states.size(), N_STATIONS);
    if (ruLogits.size() != N_STATIONS * N_RU_CLASSES)
    {
        return {};
    }
    const auto& layouts = GetLayouts();
    constexpr double NEGATIVE_INFINITY = -std::numeric_limits<double>::infinity();
    Q1RuAction bestAction;
    bestAction.objective = NEGATIVE_INFINITY;

    for (const auto& layout : layouts)
    {
        const auto slots = layout.rus.size();
        const auto maskCount = std::size_t{1} << count;
        std::vector<double> dp(maskCount, NEGATIVE_INFINITY);
        std::vector<std::vector<int>> parents(slots + 1, std::vector<int>(maskCount, -1));
        dp[0] = 0.0;
        for (std::size_t slot = 0; slot < slots; ++slot)
        {
            std::vector<double> next = dp;
            std::fill(parents[slot + 1].begin(), parents[slot + 1].end(), -2);
            const auto ruClass = GetRuClass(layout.rus[slot].GetRuType());
            for (std::size_t mask = 0; mask < maskCount; ++mask)
            {
                if (!std::isfinite(dp[mask]))
                {
                    continue;
                }
                for (std::size_t station = 0; station < count; ++station)
                {
                    if (static_cast<std::size_t>(std::popcount(mask)) >= maxStations ||
                        states[station].queue.empty() ||
                        (mask & (std::size_t{1} << station)) != 0)
                    {
                        continue;
                    }
                    const auto offset = station * N_RU_CLASSES;
                    if (!std::isfinite(ruLogits[offset]) ||
                        !std::isfinite(ruLogits[offset + ruClass]))
                    {
                        continue;
                    }
                    const auto value = dp[mask] + ruLogits[offset + ruClass] - ruLogits[offset];
                    const auto nextMask = mask | (std::size_t{1} << station);
                    if (value > next[nextMask])
                    {
                        next[nextMask] = value;
                        parents[slot + 1][nextMask] = static_cast<int>(station);
                    }
                }
            }
            dp = std::move(next);
        }

        for (std::size_t mask = 0; mask < maskCount; ++mask)
        {
            if (mask == 0 || static_cast<std::size_t>(std::popcount(mask)) > maxStations ||
                dp[mask] <= bestAction.objective)
            {
                continue;
            }
            std::vector<Q1RuAssignment> assignments;
            auto currentMask = mask;
            bool valid = true;
            for (std::size_t slot = slots; slot > 0; --slot)
            {
                const auto station = parents[slot][currentMask];
                if (station == -1)
                {
                    valid = false;
                    break;
                }
                if (station >= 0)
                {
                    assignments.push_back({static_cast<std::size_t>(station),
                                           layout.rus[slot - 1],
                                           0,
                                           0,
                                           Time{0}});
                    currentMask &= ~(std::size_t{1} << station);
                }
            }
            if (valid)
            {
                std::reverse(assignments.begin(), assignments.end());
                bestAction = {layout.code, std::move(assignments), dp[mask], Time{0}};
            }
        }
    }
    if (!std::isfinite(bestAction.objective))
    {
        return {};
    }
    return bestAction;
}

std::vector<float>
Q1RuOptimizer::BuildFeatures(const std::vector<Q1RuStationState>& states, Time guardInterval)
{
    std::vector<float> features(N_STATIONS * FEATURE_COUNT, 0.0F);
    const auto count = std::min(states.size(), N_STATIONS);
    const auto maxRate = static_cast<float>(
        HePhy::GetDataRate(11, MHz_u{20}, guardInterval, 1));
    for (std::size_t station = 0; station < count; ++station)
    {
        const auto& state = states[station];
        const auto offset = station * FEATURE_COUNT;
        features[offset] = state.queue.empty() ? 0.0F : 1.0F;
        features[offset + 1] = state.mcs / 11.0F;
        features[offset + 2] =
            HePhy::GetDataRate(state.mcs, MHz_u{20}, guardInterval, state.nss) / maxRate;
        features[offset + 3] = state.queuePackets /
                               static_cast<float>(state.queuePackets + 64U);
        features[offset + 4] = std::min(1.0F, state.queueBytes / (64.0F * 1600.0F));
        if (!state.queue.empty())
        {
            const auto meanAge =
                std::accumulate(state.queue.begin(),
                                state.queue.end(),
                                0.0,
                                [](double sum, const auto& packet) { return sum + packet.ageMs; }) /
                state.queue.size();
            features[offset + 5] = std::min(1.0, state.queue.front().ageMs / 2000.0);
            features[offset + 6] = std::min(1.0, meanAge / 2000.0);
        }
        const auto packetCount = std::min<std::size_t>(MODEL_PREFIX, state.queue.size());
        for (std::size_t packet = 0; packet < packetCount; ++packet)
        {
            features[offset + 7 + 2 * packet] =
                std::min(1.0F, state.queue[packet].payloadBytes / 1600.0F);
            features[offset + 8 + 2 * packet] =
                std::min(1.0, state.queue[packet].ageMs / 2000.0);
        }
    }
    return features;
}

} // namespace ns3
