#ifndef __ml_tCNN_h__
#define __ml_tCNN_h__


#include <ml/rhocompat.h>
#include <ml/common.h>
#include <ml/iLearner.h>

#include <rho/bNonCopyable.h>
#include <rho/eRho.h>
#include <rho/iPackable.h>
#include <rho/img/tImage.h>


namespace ml
{


class tCNN : public iPackable, public iLearner, public bNonCopyable
{
    public:

        //////////////////////////////////////////////////////////////////////
        // Constructors / resetWeights() / destructor
        //////////////////////////////////////////////////////////////////////

        /**
         * Creates a new CNN which will be subsequently trained.
         *
         * The hyper-parameters of the CNN will be described by the given
         * string.
         *
         * The description string looks like this:
         *
         *        <input:width> <input:height>
         *        <clayer:rf_width> <clayer:rf_height> <clayer:rf_stepx> <clayer:rf_stepy> <clayer::num_maps> <clayer:pool_width> <clayer:pool_height>
         *        <clayer:rf_width> <clayer:rf_height> <clayer:rf_stepx> <clayer:rf_stepy> <clayer::num_maps> <clayer:pool_width> <clayer:pool_height>
         *        ...
         *        <flayer:num_neurons>
         *        <flayer:num_neurons>
         *
         * where the "input" fields describes the size of the network's input,
         * the "clayer" fields describe a convolutional layer, and
         * the "flayer" fields describe a fully-connected layer.
         *
         * There may be no convolutional layers after the first fully-
         * connected layer.
         */
        tCNN(std::string descriptionString);

        /**
         * Constructs a CNN by restoring from a readable stream.
         */
        tCNN(iReadable* in);

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
        ~tCNN();

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
        void setLayerType(tANN::nLayerType type, u32 layerIndex);
        void setLayerType(tANN::nLayerType type);

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
        void setWeightUpRule(tANN::nWeightUpRule rule, u32 layerIndex);
        void setWeightUpRule(tANN::nWeightUpRule rule);

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
         * Shows the CNN one example. The CNN will calculate the error
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
         * Useful for generating file names for storing CNN-related data.
         */
        std::string learnerInfoString() const;

        //////////////////////////////////////////////////////////////////////
        // Getters
        //////////////////////////////////////////////////////////////////////

        /**
         * Returns the number of layers in the network. This will be one less
         * than what was specified to the constructor, because the CNN does
         * not actually use a layer for the input.
         */
        u32 getNumLayers() const;

        /**
         * Returns true if the specified layer is pooled, false if not pooled.
         */
        bool isLayerPooled(u32 layerIndex) const;

        /**
         * Returns true if the specified layer is a fully connected layer,
         * false if the specified layer is a convolutional layer.
         */
        bool isLayerFullyConnected(u32 layerIndex) const;

        /**
         * Returns the number of feature maps in the specified layer. The given
         * parameter is interpreted as the layer index where 0 is the first
         * layer above the input, and all consecutive layers are higher in the
         * network (i.e. closer to the output layer).
         */
        u32 getNumFeatureMaps(u32 layerIndex) const;

        /**
         * Returns the weights of the connections below the specified feature
         * map. The number of weights returned will equal the size of the
         * receptive field below the specified feature map.
         */
        void getWeights(u32 layerIndex, u32 mapIndex, std::vector<f64>& weights) const;

        /**
         * Returns the bias value of the specified feature map.
         *
         * The getWeights() method omits the bias of each map so that
         * the number of weights returned is equal to the dimensionality
         * of the field below. This method is how you can access the bias
         * of a map.
         */
        f64 getBias(u32 layerIndex, u32 mapIndex) const;

        /**
         * Returns the number of replicated filters in the specified
         * layer. Aka, the number of receptive fields that are imposed
         * on the input of the specified layer.
         */
        u32 getNumReplicatedFilters(u32 layerIndex) const;

        /**
         * Returns the output value of the specified filter. This will be
         * the filter's output value from the last example which was
         * pushed through this network.
         *
         * Note: Examples are not pushed through by addExample(). They are
         * only pushed through by evaluate*() and update().
         *
         * If minValue and maxValue are not NULL, they are filled
         * with the minimum and maximum possible values output by
         * the specified neuron (determined by the squashing function
         * used by that neuron).
         */
        f64 getOutput(u32 layerIndex, u32 mapIndex, u32 filterIndex,
                      f64* minValue = NULL, f64* maxValue = NULL) const;

        /**
         * Generates an image representation of the specified feature map.
         * The image shows a visual representation of the weights of
         * the connections below the specified feature map.
         *
         * This method uses ml::un_examplify() to create the image, so see
         * that method for a description of the parameters.
         *
         * The generated image is stored in 'dest'.
         */
        void getFeatureMapImage(u32 layerIndex, u32 mapIndex,
                                bool color, bool absolute,
                                img::tImage* dest) const;

        /**
         * Generates an image representation of the output of every
         * replicated filter in the specified feature map. You can
         * think of this as a representation of the transformed
         * input to the specified layer, transformed by the specified
         * feature map.
         *
         * If 'pooled', an image of the pooled output will be created.
         * Else, an image of the full output of the layer will be created.
         *
         * The generated image is stored in 'dest'.
         */
        void getOutputImage(u32 layerIndex, u32 mapIndex,
                            bool pooled,
                            img::tImage* dest) const;

        //////////////////////////////////////////////////////////////////////
        // END -- read no further
        //////////////////////////////////////////////////////////////////////

    public:

        // iPackable interface:
        void pack(iWritable* out) const;
        void unpack(iReadable* in);

    private:

        class tLayerCNN* m_layers;   // an array of layers
        u32 m_numLayers;             // number of layers

        f64 m_randWeightMin;   // used for resetWeights()
        f64 m_randWeightMax;   // ...

        std::vector<tIO> m_inputAccum;
        std::vector<tIO> m_targetAccum;
};


}    // namespace ml


#endif // __ml_tCNN_h__
