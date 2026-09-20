/*
 * SPDX-License-Identifier: GPL-2.0-only
 *
 * Reproducible downlink Wi-Fi 6 validation for the oracle-imitation scheduler.
 */

#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/network-module.h"
#include "ns3/propagation-module.h"
#include "ns3/spectrum-module.h"
#include "ns3/wifi-module.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <vector>

using namespace ns3;

namespace
{

struct ReceiverStats
{
    uint64_t packets{0};
    uint64_t bytes{0};
    std::vector<double> delaysMs;
};

std::vector<ReceiverStats> g_receiverStats;
Time g_measurementStart;
Time g_measurementStop;

void
ReceivePacket(std::size_t station,
              Ptr<const Packet> packet,
              const Address& source,
              const Address& destination)
{
    (void)source;
    (void)destination;
    const auto now = Simulator::Now();
    if (now < g_measurementStart || now > g_measurementStop)
    {
        return;
    }

    auto copy = packet->Copy();
    SeqTsHeader header;
    copy->RemoveHeader(header);
    auto& stats = g_receiverStats.at(station);
    ++stats.packets;
    stats.bytes += packet->GetSize();
    stats.delaysMs.push_back((now - header.GetTs()).ToDouble(Time::MS));
}

double
Percentile(std::vector<double> values, double quantile)
{
    if (values.empty())
    {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const auto index = static_cast<std::size_t>(
        std::ceil(quantile * static_cast<double>(values.size())) - 1.0);
    return values.at(std::min(index, values.size() - 1));
}

double
JainFairness(const std::vector<double>& values)
{
    const auto sum = std::accumulate(values.begin(), values.end(), 0.0);
    const auto squaredSum = std::inner_product(values.begin(), values.end(), values.begin(), 0.0);
    return squaredSum > 0.0 ? sum * sum / (values.size() * squaredSum) : 0.0;
}

} // namespace

int
main(int argc, char* argv[])
{
    uint32_t policy = 0;
    uint32_t stations = 9;
    uint32_t mcs = 9;
    uint32_t seed = 1;
    uint32_t payloadBytes = 1200;
    uint32_t maxScheduledStations = 9;
    uint32_t trainingStride = 1;
    double offeredMbpsPerSta = 4.0;
    double loadSkew = 0.0;
    double duration = 5.0;
    double warmup = 2.0;
    double delayWeight = 0.0;
    double delayScaleMs = 100.0;
    double exchangeOverheadUs = 100.0;
    double decisionBudgetUs = 1000.0;
    std::string modelPath = "../ml/ns3_ru_v4/throughput/scheduler_weights.bin";
    std::string trainingOutput;

    CommandLine command(__FILE__);
    command.AddValue("policy", "0 RR, 1 max backlog, 2 exact RU oracle, 3 learned RU", policy);
    command.AddValue("stations", "Number of associated and active stations", stations);
    command.AddValue("mcs", "Constant HE MCS (0-11)", mcs);
    command.AddValue("seed", "ns-3 RNG run number", seed);
    command.AddValue("payloadBytes", "UDP datagram size", payloadBytes);
    command.AddValue("maxScheduledStations", "Maximum stations/RUs per DL MU PPDU", maxScheduledStations);
    command.AddValue("trainingStride", "Write one label every this many oracle opportunities", trainingStride);
    command.AddValue("offeredMbpsPerSta", "Offered downlink load per station", offeredMbpsPerSta);
    command.AddValue("loadSkew", "Three-level per-STA load skew in [0,1)", loadSkew);
    command.AddValue("duration", "Measurement duration in seconds", duration);
    command.AddValue("warmup", "Warm-up duration in seconds", warmup);
    command.AddValue("modelPath", "Binary set-network model", modelPath);
    command.AddValue("delayWeight", "Oracle age weight; zero is throughput-only", delayWeight);
    command.AddValue("delayScaleMs", "Packet-age normalization in milliseconds", delayScaleMs);
    command.AddValue("exchangeOverheadUs", "Fixed objective overhead in microseconds", exchangeOverheadUs);
    command.AddValue("decisionBudgetUs", "Wall-clock state-to-action deadline in microseconds", decisionBudgetUs);
    command.AddValue("trainingOutput", "Optional live-state oracle JSONL output", trainingOutput);
    command.Parse(argc, argv);

    NS_ABORT_MSG_IF(stations == 0 || stations > 9, "stations must be in [1,9]");
    NS_ABORT_MSG_IF(mcs > 11, "mcs must be in [0,11]");
    NS_ABORT_MSG_IF(policy > 3, "policy must be in [0,3]");
    NS_ABORT_MSG_IF(maxScheduledStations == 0 || maxScheduledStations > 9,
                    "maxScheduledStations must be in [1,9]");
    NS_ABORT_MSG_IF(loadSkew < 0.0 || loadSkew >= 1.0, "loadSkew must be in [0,1)");
    NS_ABORT_MSG_IF(offeredMbpsPerSta <= 0.0 || duration <= 0.0, "load and duration must be positive");
    NS_ABORT_MSG_IF(decisionBudgetUs <= 0.0, "decisionBudgetUs must be positive");

    RngSeedManager::SetSeed(20260914);
    RngSeedManager::SetRun(seed);
    Config::SetDefault("ns3::WifiRemoteStationManager::RtsCtsThreshold", UintegerValue(999999));
    Config::SetDefault("ns3::WifiMac::MpduBufferSize", UintegerValue(64));

    NodeContainer stationNodes;
    stationNodes.Create(stations);
    NodeContainer apNode;
    apNode.Create(1);

    WifiHelper wifi;
    wifi.SetStandard(WIFI_STANDARD_80211ax);
    std::ostringstream dataMode;
    dataMode << "HeMcs" << mcs;
    wifi.SetRemoteStationManager("ns3::ConstantRateWifiManager",
                                 "DataMode",
                                 StringValue(dataMode.str()),
                                 "ControlMode",
                                 StringValue("OfdmRate6Mbps"));
    wifi.ConfigHeOptions("GuardInterval", TimeValue(NanoSeconds(800)));

    auto spectrumChannel = CreateObject<MultiModelSpectrumChannel>();
    spectrumChannel->AddPropagationLossModel(CreateObject<LogDistancePropagationLossModel>());
    SpectrumWifiPhyHelper phy;
    phy.SetChannel(spectrumChannel);
    phy.Set("ChannelSettings", StringValue("{0, 20, BAND_5GHZ, 0}"));

    WifiMacHelper mac;
    const Ssid ssid("q1-oracle-imitation");
    mac.SetType("ns3::StaWifiMac", "Ssid", SsidValue(ssid));
    auto stationDevices = wifi.Install(phy, mac, stationNodes);

    mac.SetMultiUserScheduler("ns3::RrMultiUserScheduler",
                              "EnableUlOfdma",
                              BooleanValue(false),
                              "NStations",
                              UintegerValue(maxScheduledStations),
                              "DlSelectionPolicy",
                              UintegerValue(policy),
                              "ModelPath",
                              StringValue(modelPath),
                              "OracleDelayWeight",
                              DoubleValue(delayWeight),
                              "OracleDelayScale",
                              TimeValue(MilliSeconds(delayScaleMs)),
                              "OracleExchangeOverhead",
                              TimeValue(MicroSeconds(exchangeOverheadUs)),
                              "TrainingOutput",
                              StringValue(trainingOutput),
                              "TrainingSampleStride",
                              UintegerValue(trainingStride));
    mac.SetType("ns3::ApWifiMac",
                "Ssid",
                SsidValue(ssid),
                "EnableBeaconJitter",
                BooleanValue(false));
    auto apDevice = wifi.Install(phy, mac, apNode);

    MobilityHelper mobility;
    auto positions = CreateObject<ListPositionAllocator>();
    for (uint32_t station = 0; station < stations; ++station)
    {
        const double angle = 2.0 * M_PI * station / stations;
        positions->Add(Vector(2.0 * std::cos(angle), 2.0 * std::sin(angle), 0.0));
    }
    positions->Add(Vector(0.0, 0.0, 0.0));
    mobility.SetPositionAllocator(positions);
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mobility.Install(stationNodes);
    mobility.Install(apNode);

    InternetStackHelper stack;
    stack.Install(stationNodes);
    stack.Install(apNode);
    Ipv4AddressHelper address;
    address.SetBase("10.1.0.0", "255.255.255.0");
    auto stationInterfaces = address.Assign(stationDevices);
    address.Assign(apDevice);

    const double clientStart = 1.0;
    g_measurementStart = Seconds(clientStart + warmup);
    g_measurementStop = Seconds(clientStart + warmup + duration);
    g_receiverStats.resize(stations);
    double totalOfferedMbps{0.0};
    for (uint32_t station = 0; station < stations; ++station)
    {
        const uint16_t port = 9000 + station;
        UdpServerHelper server(port);
        auto serverApplications = server.Install(stationNodes.Get(station));
        serverApplications.Start(Seconds(0.0));
        serverApplications.Stop(g_measurementStop + MilliSeconds(100));
        serverApplications.Get(0)->TraceConnectWithoutContext(
            "RxWithAddresses",
            MakeBoundCallback(&ReceivePacket, static_cast<std::size_t>(station)));

        UdpClientHelper client(stationInterfaces.GetAddress(station), port);
        const auto loadLevel = static_cast<int>(station % 3) - 1;
        const auto stationOfferedMbps = offeredMbpsPerSta * (1.0 + loadSkew * loadLevel);
        totalOfferedMbps += stationOfferedMbps;
        client.SetAttribute("MaxPackets", UintegerValue(0xffffffff));
        client.SetAttribute("PacketSize", UintegerValue(payloadBytes));
        client.SetAttribute(
            "Interval",
            TimeValue(Seconds(payloadBytes * 8.0 / (stationOfferedMbps * 1e6))));
        auto clientApplications = client.Install(apNode.Get(0));
        clientApplications.Start(Seconds(clientStart));
        clientApplications.Stop(g_measurementStop);
    }

    Ipv4GlobalRoutingHelper::PopulateRoutingTables();
    auto apMac = DynamicCast<ApWifiMac>(DynamicCast<WifiNetDevice>(apDevice.Get(0))->GetMac());
    NS_ASSERT(apMac);
    auto scheduler = apMac->GetObject<RrMultiUserScheduler>();
    NS_ASSERT(scheduler);
    Simulator::Stop(g_measurementStop + MilliSeconds(100));
    Simulator::Run();
    const auto decisionLatencies = scheduler->GetDecisionLatenciesUs();
    const auto inferenceLatencies = scheduler->GetInferenceLatenciesUs();
    const auto stateLatencies = scheduler->GetStateLatenciesUs();
    const auto projectionLatencies = scheduler->GetProjectionLatenciesUs();
    const auto learnedFallbacks = scheduler->GetLearnedFallbackCount();
    Simulator::Destroy();

    uint64_t receivedPackets{0};
    uint64_t receivedBytes{0};
    std::vector<double> stationThroughputs;
    std::vector<double> delays;
    for (const auto& stats : g_receiverStats)
    {
        receivedPackets += stats.packets;
        receivedBytes += stats.bytes;
        stationThroughputs.push_back(stats.bytes * 8.0 / duration / 1e6);
        delays.insert(delays.end(), stats.delaysMs.begin(), stats.delaysMs.end());
    }
    const auto throughput = receivedBytes * 8.0 / duration / 1e6;
    const auto meanDelay = delays.empty()
                               ? 0.0
                               : std::accumulate(delays.begin(), delays.end(), 0.0) / delays.size();
    const auto decisionMeanUs = decisionLatencies.empty()
                                    ? 0.0
                                    : std::accumulate(decisionLatencies.begin(),
                                                      decisionLatencies.end(),
                                                      0.0) /
                                          decisionLatencies.size();
    const auto inferenceMeanUs = inferenceLatencies.empty()
                                     ? 0.0
                                     : std::accumulate(inferenceLatencies.begin(),
                                                       inferenceLatencies.end(),
                                                       0.0) /
                                           inferenceLatencies.size();
    const auto decisionMaxUs = decisionLatencies.empty()
                                   ? 0.0
                                   : *std::max_element(decisionLatencies.begin(),
                                                       decisionLatencies.end());
    const auto inferenceMaxUs = inferenceLatencies.empty()
                                    ? 0.0
                                    : *std::max_element(inferenceLatencies.begin(),
                                                        inferenceLatencies.end());
    const auto stateMeanUs = stateLatencies.empty()
                                 ? 0.0
                                 : std::accumulate(stateLatencies.begin(), stateLatencies.end(), 0.0) /
                                       stateLatencies.size();
    const auto projectionMeanUs = projectionLatencies.empty()
                                      ? 0.0
                                      : std::accumulate(projectionLatencies.begin(),
                                                        projectionLatencies.end(),
                                                        0.0) /
                                            projectionLatencies.size();
    const auto decisionMisses = std::count_if(decisionLatencies.begin(),
                                              decisionLatencies.end(),
                                              [decisionBudgetUs](double latency) {
                                                  return latency > decisionBudgetUs;
                                              });

    std::cout << std::fixed << std::setprecision(6)
              << "RESULT policy=" << policy << " seed=" << seed << " stations=" << stations
              << " mcs=" << mcs << " offered_mbps=" << totalOfferedMbps
              << " max_scheduled_stations=" << maxScheduledStations
              << " delay_weight=" << delayWeight
              << " throughput_mbps=" << throughput << " received_packets=" << receivedPackets
              << " delay_mean_ms=" << meanDelay << " delay_p95_ms=" << Percentile(delays, 0.95)
              << " delay_p99_ms=" << Percentile(delays, 0.99)
              << " fairness=" << JainFairness(stationThroughputs)
              << " decision_count=" << decisionLatencies.size()
              << " decision_mean_us=" << decisionMeanUs
              << " decision_p95_us=" << Percentile(decisionLatencies, 0.95)
              << " decision_p99_us=" << Percentile(decisionLatencies, 0.99)
              << " decision_max_us=" << decisionMaxUs
              << " decision_budget_us=" << decisionBudgetUs
              << " decision_deadline_misses=" << decisionMisses
              << " decision_deadline_miss_rate="
              << (decisionLatencies.empty()
                      ? 0.0
                      : static_cast<double>(decisionMisses) / decisionLatencies.size())
              << " inference_mean_us=" << inferenceMeanUs
              << " inference_p95_us=" << Percentile(inferenceLatencies, 0.95)
              << " inference_p99_us=" << Percentile(inferenceLatencies, 0.99)
              << " inference_max_us=" << inferenceMaxUs
              << " state_mean_us=" << stateMeanUs
              << " state_p99_us=" << Percentile(stateLatencies, 0.99)
              << " projection_mean_us=" << projectionMeanUs
              << " projection_p99_us=" << Percentile(projectionLatencies, 0.99)
              << " learned_fallbacks=" << learnedFallbacks << std::endl;
    return 0;
}
