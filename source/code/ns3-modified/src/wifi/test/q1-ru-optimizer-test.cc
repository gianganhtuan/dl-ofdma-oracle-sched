/*
 * SPDX-License-Identifier: GPL-2.0-only
 */

#include "ns3/q1-ru-optimizer.h"
#include "ns3/test.h"

using namespace ns3;

namespace
{

WifiTxVector
MakeTxVector()
{
    WifiTxVector txVector;
    txVector.SetPreambleType(WIFI_PREAMBLE_HE_MU);
    txVector.SetChannelWidth(MHz_u{20});
    txVector.SetGuardInterval(NanoSeconds(800));
    return txVector;
}

Q1RuStationState
MakeState(uint16_t aid, double ageMs)
{
    return {aid, 6, 1, 1, 1030, 65535, {{1000, 1030, ageMs}}};
}

class Q1RuOptimizerTestCase : public TestCase
{
  public:
    Q1RuOptimizerTestCase()
        : TestCase("Q1 RU optimizer objective and legal projection")
    {
    }

  private:
    void DoRun() override
    {
        const auto txVector = MakeTxVector();
        const auto horizon = MicroSeconds(5484);
        const auto overhead = MicroSeconds(100);

        auto sparse = Q1RuOptimizer::Optimize({MakeState(1, 1.0)},
                                               txVector,
                                               WIFI_PHY_BAND_5GHZ,
                                               9,
                                               horizon,
                                               overhead,
                                               0.0,
                                               MilliSeconds(100));
        NS_TEST_ASSERT_MSG_EQ(sparse.assignments.size(), 1, "Sparse action must serve one STA");
        NS_TEST_ASSERT_MSG_EQ(sparse.ruAllocation, 192, "Sparse rate objective must use 242 tones");
        NS_TEST_ASSERT_MSG_EQ(sparse.assignments.front().ru.GetRuType(),
                              RuType::RU_242_TONE,
                              "Sparse rate objective selected the wrong RU");

        std::vector<Q1RuStationState> tied{MakeState(1, 0.0), MakeState(2, 500.0)};
        const auto rateAction = Q1RuOptimizer::Optimize(tied,
                                                        txVector,
                                                        WIFI_PHY_BAND_5GHZ,
                                                        1,
                                                        horizon,
                                                        overhead,
                                                        0.0,
                                                        MilliSeconds(100));
        const auto delayAction = Q1RuOptimizer::Optimize(tied,
                                                         txVector,
                                                         WIFI_PHY_BAND_5GHZ,
                                                         1,
                                                         horizon,
                                                         overhead,
                                                         2.0,
                                                         MilliSeconds(100));
        NS_TEST_ASSERT_MSG_EQ(rateAction.assignments.front().stationIndex,
                              0,
                              "Deterministic throughput tie break changed");
        NS_TEST_ASSERT_MSG_EQ(delayAction.assignments.front().stationIndex,
                              1,
                              "Age weighting did not prioritize the older queue");

        std::vector<float> logits(Q1RuOptimizer::N_STATIONS *
                                      Q1RuOptimizer::N_RU_CLASSES,
                                  -10.0F);
        logits[0] = 0.0F;
        logits[4] = 10.0F;
        const auto decoded = Q1RuOptimizer::Decode({MakeState(1, 1.0)}, logits, 9);
        NS_TEST_ASSERT_MSG_EQ(decoded.assignments.size(), 1, "Projection returned no action");
        NS_TEST_ASSERT_MSG_EQ(decoded.ruAllocation, 192, "Projection selected wrong partition");
        NS_TEST_ASSERT_MSG_EQ(decoded.assignments.front().ru.GetRuType(),
                              RuType::RU_242_TONE,
                              "Projection selected wrong RU type");
    }
};

class Q1RuOptimizerTestSuite : public TestSuite
{
  public:
    Q1RuOptimizerTestSuite()
        : TestSuite("q1-ru-optimizer", Type::UNIT)
    {
        AddTestCase(new Q1RuOptimizerTestCase, TestCase::Duration::QUICK);
    }
};

static Q1RuOptimizerTestSuite g_q1RuOptimizerTestSuite;

} // namespace
