#ifndef __ml_tANN_h__
#define __ml_tANN_h__


#include <ml/rhocompat.h>
#include <ml/common.h>
#include <ml/iLearner.h>

#include <rho/bNonCopyable.h>
#include <rho/eRho.h>
#include <rho/iPackable.h>
#include <rho/img/tImage.h>


namespace ml
{


class tANN : public iPackable, public iLearner, public bNonCopyable
{
    public:

        //////////////////////////////////////////////////////////////////////
        // Enumerations used within tANN
        //////////////////////////////////////////////////////////////////////

        /**
         * Each layer of the ANN has a certain type. That type defines
         * which squashing function is used on the neurons in that layer.
         */
        enum nLayerType {
            kLayerTypeLogistic      = 0, // the logistic function
            kLayerTypeHyperbolic    = 1, // the hyperbolic tangent function
            kLayerTypeSoftmax       = 2, // a softmax group
            kLayerTypeMax           = 3  // marks the max of this enum (do not use)
        };

        /**
         * Each layer also has a weight update rule. It determines how the
         * gradient is used to update that layer's incoming weights during
         * training.
         */
        enum nWeightUpRule {
            kWeightUpRuleNone              = 0,  // no changes will be made to the weights
            kWeightUpRuleFixedLearningRate = 1,  // the standard fixed learning rate method
            kWeightUpRuleMomentum          = 2,  // the momentum learning rate method
            kWeightUpRuleAdaptiveRates     = 3,  // the adaptive learning rates method (for full- or large-batch)
            kWeightUpRuleRPROP             = 4,  // the rprop full-batch method
            kWeightUpRuleRMSPROP           = 5,  // the rmsprop mini-batch method (a mini-batch version of rprop)
            kWeightUpRuleARMS              = 6,  // the adaptive rmsprop method
            kWeightUpRuleMax               = 7   // marks the max of this enum (do not use)
        };

        //////////////////////////////////////////////////////////////////////
        // Constructors / resetWeights() / destructor
        //////////////////////////////////////////////////////////////////////

        /**
         * Reads the ANN from an input stream. This is useful for
         * restoring an ANN from disk for more training, or for using
         * a pre-trained ANN in an application.
         */
        tANN(iReadable* in);

        /**
         * Creates a new ANN which will be subsequently trained.
         *
         * The size of each layer will be specified by the 'layerSizes'
         * vector. "layerSizes.front()" will be the dimensionality of the
         * ANN's input. "layerSizes.back()" will be the dimensionality of
         * the ANN's output. All other specified layers will be hidden
         * layers in the ANN, meaning that there are "layerSizes.size()-2"
         * hidden layers. The network will not actually create a layer
         * for the input, so the number of layers in the network will be
         * equal to "layerSizes.size()-1". The layer immediately above the
         * input will be called the "bottom layer". The layer that represents
         * the output will be called the "top layer".
         *
         * The network's weights are initialized randomly over the uniform
         * range ['randWeightMin', 'randWeightMax'].
         *
         * By default:
         *     1. The network's layer type is set as kLayerTypeHyperbolic
         *        for all layers.
         *     2. Every layer will have 'normalizeLayerInput' set to true.
         *     3. The network will NOT be ready to learn; that is, all the
         *        layers' weight update rules will be set to kWeightUpRuleNone.
         *        Use the methods below to specify how the network should
         *        do its learning before you begin training.
         */
        tANN(std::vector<u32> layerSizes,
             f64 randWeightMin = -1.0,
             f64 randWeightMax = 1.0);

        /**
         * Resets the weighs in the network to the initial random weights
         * set by the above constructor.
         *
         * This is useful when you need to train the same network several
         * times with varying training sets (e.g. for ten-fold cross-
         * validation, or for exploring how the amount of training data
         * affects performance).
         */
        void resetWeights();

        /**
         * Destructor...
         */
        ~tANN();

        //////////////////////////////////////////////////////////////////////
        // Network configuration -- do this before training!
        //////////////////////////////////////////////////////////////////////

        /**
         * Sets the layer type of one layer (first method) or
         * all layers (second method). The layer type determines
         * which squashing function is used on the neurons in a
         * layer.
         *
         * Note: You may only specify kLayerTypeSoftmax for the
         *       top layer.
         */
        void setLayerType(nLayerType type, u32 layerIndex);
        void setLayerType(nLayerType type);

        /**
         * Sets whether or not a layer normalizes its input wrt
         * the number of inputs. Doing this keeps the total input
         * to a neuron from being very large early in training, which
         * is common for neurons with many inputs, and which causes
         * the neuron to be over-saturated making learning slow due
         * to small gradients on the plateaus of the squashing function.
         * This method turns this feature on or off for one layer
         * (first method) or all layers (second method).
         */
        void setNormalizeLayerInput(bool on, u32 layerIndex);
        void setNormalizeLayerInput(bool on);

        /**
         * Sets the weight update rule for one layer (first method)
         * or all layers (second method).
         *
         * For each layer, you must also setup the learning parameters
         * associated with the weight update rule for that layer.
         * Below describes the parameters needed for each rule:
         *
         *    - kWeightUpRuleNone
         *         -- no extra parameters needed
         *
         *    - kWeightUpRuleFixedLearningRate
         *         -- requires setAlpha()
         *
         *    - kWeightUpRuleMomentum
         *         -- requires setAlpha() and setViscosity()
         *
         *    - kWeightUpRuleAdaptiveRates
         *         -- requires setAlpha()
         *         -- requires using full- or large-batch learning
         *
         *    - kWeightUpRuleRPROP
         *         -- no extra parameters needed
         *         -- requires full-batch learning
         *
         *    - kWeightUpRuleRMSPROP
         *         -- requires setAlpha()
         *         -- this is a mini-batch version of the rprop method
         *
         *    - kWeightUpRuleARMS
         *         -- requires setAlpha()
         *         -- very similar to RMSPROP, but has an addaptive alpha
         */
        void setWeightUpRule(nWeightUpRule rule, u32 layerIndex);
        void setWeightUpRule(nWeightUpRule rule);

        /**
         * Sets the alpha parameter for one layer (first method) or
         * all layers (second method). The alpha parameter is the
         * "fixed learning rate" parameter, used when the weight update
         * rule is kWeightUpRuleFixedLearningRate.
         *
         * This parameter is also used when the weight update rule
         * is kWeightUpRuleMomentum.
         *
         * This parameter is also used when the weight update rule
         * is kWeightUpRuleAdaptiveRates for the "base rate".
         * Note: If you use kWeightUpRuleAdaptiveRates, you must
         * use full- or large-batch learning.
         *
         * This parameter is also used when the weight update rule
         * is kWeightUpRuleRMSPROP or kWeightUpRuleARMS.
         */
        void setAlpha(f64 alpha, u32 layerIndex);
        void setAlpha(f64 alpha);

        /**
         * Sets the viscosity of the network's weight velocities when using
         * the momentum weight update rule (kWeightUpRuleMomentum).
         * Sets the viscosity of one layer (first method) or all layers
         * (second method).
         */
        void setViscosity(f64 viscosity, u32 layerIndex);
        void setViscosity(f64 viscosity);

        //////////////////////////////////////////////////////////////////////
        // Training -- this is the iLearner interface
        //////////////////////////////////////////////////////////////////////

        /**
         * Shows the ANN one example. The ANN will calculate the error
         * gradients for each weight in the network, then add those gradients
         * to the accumulated error gradients for each weight.
         *
         * The network's weights will not be updated by this method,
         * and this example will not actually be pushed through the
         * network at this point either.
         */
        void addExample(const tIO& input, const tIO& target);

        /**
         * Updates the network's weights using the accumulated error
         * gradients. Updates are made according to the current weight
         * update rule for each layer.
         *
         * The accumulated error gradients are reset by this method.
         */
        void update();

        /**
         * Uses the network's current weights to evaluate the given input.
         */
        void evaluate(const tIO& input, tIO& output) const;

        /**
         * Uses the network's current weights to evaluate the given input.
         *
         * This is more efficient than calling the above version of evaluate()
         * over-and-over.
         */
        void evaluateBatch(const std::vector<tIO>& inputs,
                                 std::vector<tIO>& outputs) const;

        /**
         * Same as above, but using iterators.
         */
        void evaluateBatch(std::vector<tIO>::const_iterator inputStart,
                           std::vector<tIO>::const_iterator inputEnd,
                           std::vector<tIO>::iterator outputStart) const;

        /**
         * Calculates the standard squared error if the output layer of
         * the network is logistic or hyperbolic.
         * Calculates the cross-entropy cost if the output layer of the
         * network is a softmax group.
         */
        f64 calculateError(const tIO& output, const tIO& target);

        /**
         * Calculates the average standard squared error if the output layer of
         * the network is logistic or hyperbolic.
         * Calculates the average cross-entropy cost if the output layer of the
         * network is a softmax group.
         */
        f64 calculateError(const std::vector<tIO>& outputs,
                           const std::vector<tIO>& targets);

        /**
         * Resets the learner to its initial state.
         * (This just calls resetWeights().)
         */
        void reset();

        /**
         * Prints the network's configuration in a readable format.
         */
        void printLearnerInfo(std::ostream& out) const;

        /**
         * Returns a single-line version of printLearnerInfo().
         *
         * Useful for generating file names for storing ANN-related data.
         */
        std::string learnerInfoString() const;

        //////////////////////////////////////////////////////////////////////
        // Getters
        //////////////////////////////////////////////////////////////////////

        /**
         * Returns the number of layers in the network. This will be one less
         * than what was specified to the constructor, because the ANN does
         * not actually use a layer for the input.
         */
        u32 getNumLayers() const;

        /**
         * Returns the number of neurons in the specified layer. The given
         * parameter is interpreted as the layer index where 0 is the first
         * layer above the input, and all consecutive layers are higher in the
         * network (i.e. closer to the output layer).
         */
        u32 getNumNeuronsInLayer(u32 layerIndex) const;

        /**
         * Returns the weights of the connections below the specified neuron.
         * The number of weights returned will equal the number of neurons
         * in the layer below the specified neuron. If the specified layer is 0,
         * the number of weights returned will equal the dimensionality of the
         * input.
         */
        void getWeights(u32 layerIndex, u32 neuronIndex, std::vector<f64>& weights) const;

        /**
         * Returns the bias value of the specified neuron.
         *
         * The getWeights() method omits the bias of each neuron so that
         * the number of weights returned is equal to the dimensionality
         * of the layer below. This method is how you can access the bias
         * of a neuron.
         */
        f64 getBias(u32 layerIndex, u32 neuronIndex) const;

        /**
         * Returns the output value of the specified neuron. This will be
         * the neuron's output value from the last example which was
         * pushed through this network.
         *
         * Note: Examples are not pushed through by addExample(). They are
         * only pushed through by evaluate*() and update().
         */
        f64 getOutput(u32 layerIndex, u32 neuronIndex) const;

        /**
         * Generates an image representation of the specified neuron.
         * The image shows a visual representation of the weights of
         * the connections below the specified neuron, as well as the
         * neuron's most recent output value.
         *
         * This method uses ml::un_examplify() to create the image
         * from the weight under the specified neuron, so see that
         * method for an explanation of the parameters.
         *
         * The generated image is stored in 'dest'.
         */
        void getImage(u32 layerIndex, u32 neuronIndex,
                      bool color, u32 width, bool absolute,
                      img::tImage* dest) const;

        //////////////////////////////////////////////////////////////////////
        // END -- read no further
        //////////////////////////////////////////////////////////////////////

    public:

        // iPackable interface:
        void pack(iWritable* out) const;
        void unpack(iReadable* in);

    private:

        class tLayer* m_layers;  // an array of layers
        u32 m_numLayers;         // number of layers

        f64 m_randWeightMin;     // used for resetWeights()
        f64 m_randWeightMax;     // used for resetWeights()

        std::vector<tIO> m_inputAccum;
        std::vector<tIO> m_targetAccum;
};


}    // namespace ml


#endif // __ml_tANN_h__
