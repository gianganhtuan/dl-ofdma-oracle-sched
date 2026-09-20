/*
 * SPDX-License-Identifier: GPL-2.0-only
 */

#include "q1-policy-model.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>

namespace ns3
{

namespace
{
constexpr std::size_t N_STATIONS = 9;
constexpr std::size_t N_SELECTION_TENSORS = 12;
constexpr std::size_t N_RU_TENSORS = 14;
constexpr std::size_t N_RU_CLASSES = 5;
} // namespace

bool
Q1PolicyModel::Load(const std::string& path)
{
    m_featureCount = 0;
    m_hiddenSize = 0;
    m_version = 0;
    m_tensors.clear();

    std::ifstream stream(path, std::ios::binary);
    std::array<char, 8> magic{};
    std::array<uint32_t, 5> dimensions{};
    if (!stream.read(magic.data(), magic.size()) ||
        !stream.read(reinterpret_cast<char*>(dimensions.data()), sizeof(dimensions)))
    {
        return false;
    }
    const auto identifier = std::string(magic.data(), 7);
    if ((identifier != "Q1SET01" && identifier != "Q1SET02") ||
        dimensions[0] != (identifier == "Q1SET02" ? 2U : 1U) || dimensions[1] == 0 ||
        dimensions[1] > 10000 ||
        dimensions[2] == 0 || dimensions[2] > 10000)
    {
        return false;
    }

    const auto tensorCount = identifier == "Q1SET02" ? N_RU_TENSORS : N_SELECTION_TENSORS;
    std::vector<Tensor> tensors(tensorCount);
    for (auto& tensor : tensors)
    {
        uint64_t count{0};
        if (!stream.read(reinterpret_cast<char*>(&count), sizeof(count)) || count > 100000000)
        {
            return false;
        }
        tensor.values.resize(count);
        if (!stream.read(reinterpret_cast<char*>(tensor.values.data()), count * sizeof(float)))
        {
            return false;
        }
        if (!std::all_of(tensor.values.begin(), tensor.values.end(), [](float value) {
                return std::isfinite(value);
            }))
        {
            return false;
        }
    }

    const auto features = static_cast<std::size_t>(dimensions[1]);
    const auto hidden = static_cast<std::size_t>(dimensions[2]);
    const std::array<std::size_t, N_RU_TENSORS> expectedSizes = {
        hidden * features,
        hidden,
        hidden * hidden,
        hidden,
        hidden * 2 * hidden,
        hidden,
        hidden * 2 * hidden,
        hidden,
        hidden * hidden,
        hidden,
        hidden,
        1,
        N_RU_CLASSES * hidden,
        N_RU_CLASSES};
    for (std::size_t index = 0; index < tensors.size(); ++index)
    {
        if (tensors[index].values.size() != expectedSizes[index])
        {
            return false;
        }
    }

    m_featureCount = features;
    m_hiddenSize = hidden;
    m_version = dimensions[0];
    m_tensors = std::move(tensors);
    return true;
}

bool
Q1PolicyModel::IsLoaded() const
{
    return !m_tensors.empty();
}

std::vector<float>
Q1PolicyModel::Dense(const std::vector<float>& input,
                     const Tensor& weight,
                     const Tensor& bias,
                     std::size_t inputSize,
                     std::size_t outputSize,
                     bool relu)
{
    std::vector<float> output(outputSize, 0.0F);
    for (std::size_t row = 0; row < outputSize; ++row)
    {
        float value = bias.values[row];
        const auto offset = row * inputSize;
        for (std::size_t col = 0; col < inputSize; ++col)
        {
            value += weight.values[offset + col] * input[col];
        }
        output[row] = relu ? std::max(0.0F, value) : value;
    }
    return output;
}

std::vector<float>
Q1PolicyModel::PredictSelection(const std::vector<float>& features) const
{
    if (!IsLoaded() || features.size() != N_STATIONS * m_featureCount)
    {
        return {};
    }

    std::vector<std::vector<float>> local(N_STATIONS);
    std::vector<float> mean(m_hiddenSize, 0.0F);
    std::vector<float> maximum(m_hiddenSize, -std::numeric_limits<float>::infinity());
    std::size_t activeCount{0};

    for (std::size_t sta = 0; sta < N_STATIONS; ++sta)
    {
        const auto begin = features.begin() + sta * m_featureCount;
        std::vector<float> row(begin, begin + m_featureCount);
        local[sta] = Dense(row, m_tensors[0], m_tensors[1], m_featureCount, m_hiddenSize, true);
        local[sta] = Dense(local[sta],
                           m_tensors[2],
                           m_tensors[3],
                           m_hiddenSize,
                           m_hiddenSize,
                           true);
        if (row[0] < 0.5F)
        {
            std::fill(local[sta].begin(), local[sta].end(), 0.0F);
            continue;
        }
        ++activeCount;
        for (std::size_t i = 0; i < m_hiddenSize; ++i)
        {
            mean[i] += local[sta][i];
            maximum[i] = std::max(maximum[i], local[sta][i]);
        }
    }

    if (activeCount == 0)
    {
        return std::vector<float>(N_STATIONS, 0.0F);
    }
    for (auto& value : mean)
    {
        value /= activeCount;
    }

    std::vector<float> pooled;
    pooled.reserve(2 * m_hiddenSize);
    pooled.insert(pooled.end(), mean.begin(), mean.end());
    pooled.insert(pooled.end(), maximum.begin(), maximum.end());
    auto context = Dense(pooled,
                         m_tensors[4],
                         m_tensors[5],
                         2 * m_hiddenSize,
                         m_hiddenSize,
                         true);

    std::vector<float> probabilities(N_STATIONS, 0.0F);
    for (std::size_t sta = 0; sta < N_STATIONS; ++sta)
    {
        std::vector<float> combined;
        combined.reserve(2 * m_hiddenSize);
        combined.insert(combined.end(), local[sta].begin(), local[sta].end());
        combined.insert(combined.end(), context.begin(), context.end());
        auto hidden = Dense(combined,
                            m_tensors[6],
                            m_tensors[7],
                            2 * m_hiddenSize,
                            m_hiddenSize,
                            true);
        hidden = Dense(hidden,
                       m_tensors[8],
                       m_tensors[9],
                       m_hiddenSize,
                       m_hiddenSize,
                       true);
        const auto logit = Dense(hidden,
                                 m_tensors[10],
                                 m_tensors[11],
                                 m_hiddenSize,
                                 1,
                                 false)[0];
        probabilities[sta] = 1.0F / (1.0F + std::exp(-logit));
    }
    return probabilities;
}

std::vector<float>
Q1PolicyModel::PredictRuLogits(const std::vector<float>& features) const
{
    if (!IsLoaded() || m_version < 2 || features.size() != N_STATIONS * m_featureCount)
    {
        return {};
    }

    std::vector<std::vector<float>> local(N_STATIONS);
    std::vector<float> mean(m_hiddenSize, 0.0F);
    std::vector<float> maximum(m_hiddenSize, -std::numeric_limits<float>::infinity());
    std::size_t activeCount{0};
    for (std::size_t sta = 0; sta < N_STATIONS; ++sta)
    {
        const auto begin = features.begin() + sta * m_featureCount;
        std::vector<float> row(begin, begin + m_featureCount);
        local[sta] = Dense(row, m_tensors[0], m_tensors[1], m_featureCount, m_hiddenSize, true);
        local[sta] = Dense(local[sta],
                           m_tensors[2],
                           m_tensors[3],
                           m_hiddenSize,
                           m_hiddenSize,
                           true);
        if (row[0] < 0.5F)
        {
            std::fill(local[sta].begin(), local[sta].end(), 0.0F);
            continue;
        }
        ++activeCount;
        for (std::size_t feature = 0; feature < m_hiddenSize; ++feature)
        {
            mean[feature] += local[sta][feature];
            maximum[feature] = std::max(maximum[feature], local[sta][feature]);
        }
    }
    if (activeCount == 0)
    {
        return std::vector<float>(N_STATIONS * N_RU_CLASSES, 0.0F);
    }
    for (auto& value : mean)
    {
        value /= activeCount;
    }

    std::vector<float> pooled;
    pooled.reserve(2 * m_hiddenSize);
    pooled.insert(pooled.end(), mean.begin(), mean.end());
    pooled.insert(pooled.end(), maximum.begin(), maximum.end());
    const auto context = Dense(pooled,
                               m_tensors[4],
                               m_tensors[5],
                               2 * m_hiddenSize,
                               m_hiddenSize,
                               true);

    std::vector<float> logits(N_STATIONS * N_RU_CLASSES, 0.0F);
    for (std::size_t sta = 0; sta < N_STATIONS; ++sta)
    {
        std::vector<float> combined;
        combined.reserve(2 * m_hiddenSize);
        combined.insert(combined.end(), local[sta].begin(), local[sta].end());
        combined.insert(combined.end(), context.begin(), context.end());
        auto hidden = Dense(combined,
                            m_tensors[6],
                            m_tensors[7],
                            2 * m_hiddenSize,
                            m_hiddenSize,
                            true);
        hidden = Dense(hidden,
                       m_tensors[8],
                       m_tensors[9],
                       m_hiddenSize,
                       m_hiddenSize,
                       true);
        const auto row = Dense(hidden,
                               m_tensors[12],
                               m_tensors[13],
                               m_hiddenSize,
                               N_RU_CLASSES,
                               false);
        std::copy(row.begin(), row.end(), logits.begin() + sta * N_RU_CLASSES);
    }
    return logits;
}

} // namespace ns3
