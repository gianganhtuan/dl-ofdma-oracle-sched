/*
 * SPDX-License-Identifier: GPL-2.0-only
 */

#ifndef Q1_RU_OPTIMIZER_H
#define Q1_RU_OPTIMIZER_H

#include "he-ru.h"

#include "ns3/nstime.h"
#include "ns3/wifi-phy-common.h"
#include "ns3/wifi-tx-vector.h"

#include <cstdint>
#include <vector>

namespace ns3
{

/** A queued packet as observed by the AP at a scheduling opportunity. */
struct Q1RuPacketState
{
    uint32_t payloadBytes{0}; //!< MSDU bytes used in the scheduling utility
    uint32_t mpduBytes{0};    //!< MPDU bytes used for PHY duration calculation
    double ageMs{0.0};        //!< Time elapsed since enqueue at the AP MAC
};

/** Per-station state supplied to the oracle and learned policy. */
struct Q1RuStationState
{
    uint16_t aid{0};                   //!< Association identifier
    uint8_t mcs{0};                    //!< HE MCS selected by the station manager
    uint8_t nss{1};                    //!< Number of spatial streams
    uint32_t queuePackets{0};          //!< Complete queue occupancy in packets
    uint32_t queueBytes{0};            //!< Complete queue occupancy in bytes
    uint32_t maxAmpduBytes{65535};     //!< Negotiated/configured A-MPDU byte limit
    std::vector<Q1RuPacketState> queue; //!< FIFO prefix used by the optimizer
};

/** Service selected for one station in an HE MU PPDU. */
struct Q1RuAssignment
{
    std::size_t stationIndex{0}; //!< Row in the state/feature matrix
    HeRu::RuSpec ru;             //!< Standard-compliant physical RU
    std::size_t packetCount{0};  //!< Queue prefix represented by the objective
    uint32_t payloadBytes{0};    //!< Payload bytes represented by the objective
    Time duration{0};            //!< Predicted PHY duration for this user
};

/** A complete legal 20 MHz HE RU allocation. */
struct Q1RuAction
{
    uint8_t ruAllocation{192};              //!< HE-SIG-B RU_ALLOCATION value
    std::vector<Q1RuAssignment> assignments; //!< Assigned stations and RUs
    double objective{0.0};                   //!< Oracle utility or model score
    Time ppduDuration{0};                    //!< Predicted common PPDU duration
};

/**
 * Optimize or decode 20 MHz HE OFDMA allocations for at most nine stations.
 *
 * The oracle enumerates every standard RU partition and candidate duration threshold, then solves
 * the station assignment exactly with dynamic programming. Each station/RU pair uses the longest
 * FIFO prefix that fits the configured PPDU horizon, matching the subsequent ns-3 A-MPDU request.
 * The objective is age-weighted payload divided by PHY duration plus fixed exchange overhead.
 */
class Q1RuOptimizer
{
  public:
    static constexpr std::size_t N_STATIONS{9}; //!< Fixed set-network cardinality
    static constexpr std::size_t MAX_PREFIX{64}; //!< Controlled BlockAck window size
    static constexpr std::size_t MODEL_PREFIX{16}; //!< FIFO prefix exposed to online inference
    static constexpr std::size_t FEATURE_COUNT{39}; //!< Seven scalars and 16 size/age pairs
    static constexpr std::size_t N_RU_CLASSES{5}; //!< none, 26, 52, 106, 242

    /**
     * Compute the objective-maximizing allocation.
     *
     * @param states live AP queue state
     * @param txVector HE MU TXVECTOR template
     * @param band PHY frequency band
     * @param maxStations maximum number of assigned stations
     * @param maxDuration maximum permitted PPDU duration
     * @param exchangeOverhead fixed contention/control overhead in the objective
     * @param delayWeight age weight; zero produces the throughput-only objective
     * @param delayScale normalization constant for packet age
     * @return best legal action, or an empty action when no packet can be served
     */
    static Q1RuAction Optimize(const std::vector<Q1RuStationState>& states,
                               const WifiTxVector& txVector,
                               WifiPhyBand band,
                               std::size_t maxStations,
                               Time maxDuration,
                               Time exchangeOverhead,
                               double delayWeight,
                               Time delayScale);

    /**
     * Project per-station RU logits onto the same legal action space used by the oracle.
     *
     * @param states live AP queue state
     * @param ruLogits row-major logits with five entries per station
     * @param maxStations maximum number of assigned stations
     * @return highest-scoring legal action
     */
    static Q1RuAction Decode(const std::vector<Q1RuStationState>& states,
                             const std::vector<float>& ruLogits,
                             std::size_t maxStations);

    /**
     * Build the normalized row-major feature matrix used for training and inference.
     *
     * @param states live AP queue state
     * @param guardInterval configured HE guard interval
     * @return nine rows of FEATURE_COUNT features
     */
    static std::vector<float> BuildFeatures(const std::vector<Q1RuStationState>& states,
                                            Time guardInterval);

    /**
     * Convert an RU type to the model class index.
     *
     * @param type RU type
     * @return class in [1, 4], or zero for unsupported types
     */
    static uint8_t GetRuClass(RuType type);
};

} // namespace ns3

#endif // Q1_RU_OPTIMIZER_H
