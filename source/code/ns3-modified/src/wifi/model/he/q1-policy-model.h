/*
 * SPDX-License-Identifier: GPL-2.0-only
 */

#ifndef Q1_POLICY_MODEL_H
#define Q1_POLICY_MODEL_H

#include <string>
#include <cstdint>
#include <vector>

namespace ns3
{

/**
 * @ingroup wifi
 * Lightweight inference for the permutation-equivariant oracle-imitation model.
 */
class Q1PolicyModel
{
  public:
    /**
     * Load a binary model exported by ml/export_q1_binary.py.
     *
     * @param path path to the exported model
     * @return true if the complete model is valid and was loaded
     */
    bool Load(const std::string& path);

    /**
     * Return one selection probability per STA for a row-major input.
     *
     * @param features flattened feature matrix for nine STAs
     * @return one selection probability per STA, or an empty vector on invalid input
     */
    std::vector<float> PredictSelection(const std::vector<float>& features) const;

    /**
     * Return five RU-class logits per STA in row-major order.
     *
     * @param features flattened feature matrix for nine STAs
     * @return logits for none, 26, 52, 106 and 242 tones, or an empty vector
     */
    std::vector<float> PredictRuLogits(const std::vector<float>& features) const;

    /**
     * Check whether a model has been loaded successfully.
     *
     * @return true if a model is available
     */
    bool IsLoaded() const;

  private:
    struct Tensor
    {
        std::vector<float> values; //!< Flattened row-major values
    };

    /**
     * Evaluate a dense layer.
     *
     * @param input layer input
     * @param weight row-major weight matrix
     * @param bias bias vector
     * @param inputSize number of input features
     * @param outputSize number of output features
     * @param relu whether to apply ReLU
     * @return layer output
     */
    static std::vector<float> Dense(const std::vector<float>& input,
                                    const Tensor& weight,
                                    const Tensor& bias,
                                    std::size_t inputSize,
                                    std::size_t outputSize,
                                    bool relu);

    std::size_t m_featureCount{0}; //!< Features per STA
    std::size_t m_hiddenSize{0};  //!< Hidden-layer width
    uint32_t m_version{0};        //!< Binary format version
    std::vector<Tensor> m_tensors; //!< Ordered model tensors
};

} // namespace ns3

#endif // Q1_POLICY_MODEL_H
