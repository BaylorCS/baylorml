#if __linux__
#pragma GCC optimize 3
#endif

#include <ml/tANN.h>

#include <rho/algo/tLCG.h>

#if NDEBUG
#include "Eigen/Core"
#else
#define NDEBUG 1             // <-- comment-out these two lines if you need to debug tANN
#include "Eigen/Core"        //     or tCNN, especially if your program is crashing
#undef NDEBUG                // <-- somewhere inside Eigen doing so will help a lot
#endif

#include <cassert>
#include <iomanip>
#include <sstream>

using std::string;
using std::endl;


namespace ml
{


static
string s_layerTypeToString(tANN::nLayerType type)
{
    switch (type)
    {
        case tANN::kLayerTypeLogistic:
            return "logistic";
        case tANN::kLayerTypeHyperbolic:
            return "hyperbolic";
        case tANN::kLayerTypeSoftmax:
            return "softmax";
        default:
            assert(false);
    }
}


static
char s_layerTypeToChar(tANN::nLayerType type)
{
    switch (type)
    {
        case tANN::kLayerTypeLogistic:
            return 'l';
        case tANN::kLayerTypeHyperbolic:
            return 'h';
        case tANN::kLayerTypeSoftmax:
            return 's';
        default:
            assert(false);
    }
}


static
string s_weightUpRuleToString(tANN::nWeightUpRule rule)
{
    switch (rule)
    {
        case tANN::kWeightUpRuleNone:
            return "none";
        case tANN::kWeightUpRuleFixedLearningRate:
            return "fixedrate";
        case tANN::kWeightUpRuleMomentum:
            return "mmntm";
        case tANN::kWeightUpRuleAdaptiveRates:
            return "adptvrates";
        case tANN::kWeightUpRuleRPROP:
            return "rprop";
        case tANN::kWeightUpRuleRMSPROP:
            return "rmsprop";
        case tANN::kWeightUpRuleARMS:
            return "arms";
        default:
            assert(false);
    }
}


static
char s_weightUpRuleToChar(tANN::nWeightUpRule rule)
{
    switch (rule)
    {
        case tANN::kWeightUpRuleNone:
            return 'n';
        case tANN::kWeightUpRuleFixedLearningRate:
            return 'f';
        case tANN::kWeightUpRuleMomentum:
            return 'm';
        case tANN::kWeightUpRuleAdaptiveRates:
            return 'a';
        case tANN::kWeightUpRuleRPROP:
            return 'r';
        case tANN::kWeightUpRuleRMSPROP:
            return 'R';
        case tANN::kWeightUpRuleARMS:
            return 'A';
        default:
            assert(false);
    }
}


static
f64 s_squash(f64 val, tANN::nLayerType type)
{
    switch (type)
    {
        case tANN::kLayerTypeLogistic:
            return logistic_function(val);
        case tANN::kLayerTypeHyperbolic:
            return hyperbolic_function(val);
        case tANN::kLayerTypeSoftmax:
            // Softmax layers must be handled specially
        default:
            assert(false);
    }
}


static
f64 s_derivative_of_squash(f64 val, tANN::nLayerType type)
{
    switch (type)
    {
        case tANN::kLayerTypeLogistic:
            return derivative_of_logistic_function(val);
        case tANN::kLayerTypeHyperbolic:
            return derivative_of_hyperbolic_function(val);
        case tANN::kLayerTypeSoftmax:
            // Softmax layers must be handled specially
        default:
            assert(false);
    }
}


static
f64 s_squash_min(tANN::nLayerType type)
{
    switch (type)
    {
        case tANN::kLayerTypeLogistic:
            return logistic_function_min();
        case tANN::kLayerTypeHyperbolic:
            return hyperbolic_function_min();
        case tANN::kLayerTypeSoftmax:
            return 0.0;
        default:
            assert(false);
    }
}


static
f64 s_squash_max(tANN::nLayerType type)
{
    switch (type)
    {
        case tANN::kLayerTypeLogistic:
            return logistic_function_max();
        case tANN::kLayerTypeHyperbolic:
            return hyperbolic_function_max();
        case tANN::kLayerTypeSoftmax:
            return 1.0;
        default:
            assert(false);
    }
}


class tSquashFunc
{
    public:

        tSquashFunc(tANN::nLayerType type)
            : m_type(type) { }

        f64 operator()(f64 val) const { return s_squash(val, m_type); }

    private:

        tANN::nLayerType m_type;
};


class tDirSquashFunc
{
    public:

        tDirSquashFunc(tANN::nLayerType type)
            : m_type(type) { }

        f64 operator()(f64 val) const { return s_derivative_of_squash(val, m_type); }

    private:

        tANN::nLayerType m_type;
};


class tExpFunc
{
    public:

        f64 operator()(f64 val) const { return std::min(std::exp(val), 1e100); }
};


class t_RMSPROP_wUpdate
{
    public:

        f64 operator()(f64 dw_accum, f64 dw_accum_avg) const
        {
            return (dw_accum_avg > 0.0) ? (dw_accum / std::sqrt(dw_accum_avg)) : (0.0);
        }
};


typedef Eigen::Matrix<fml,Eigen::Dynamic,Eigen::Dynamic> Mat;


class tLayer : public bNonCopyable
{ public:

    /////////////////////////////////////////////////////////////////////////////////////
    // Connection state
    /////////////////////////////////////////////////////////////////////////////////////

    Mat output;   // same as 'a', but contains a row of 1's at the bottom

    Mat a;    // the output of each neuron (the squashed values)
    Mat A;    // the accumulated input of each neuron (the pre-squashed values)

    Mat da;    // dE/da -- the error gradient wrt the output of each neuron
    Mat dA;    // dE/dA -- the error gradient wrt the accumulated input of each neuron

    Mat w;    // the weights connecting to the layer below (i.e. the previous layer)

    Mat dw_accum;    // dE/dw -- the error gradient wrt each weight

    Mat vel;             // the weight velocities (for when using momentum)
    Mat dw_accum_avg;    // moving average of dE/dw (for when using rmsprop)

    /////////////////////////////////////////////////////////////////////////////////////
    // Behavioral state -- defines the squashing function and derivative calculations
    /////////////////////////////////////////////////////////////////////////////////////

    tANN::nLayerType layerType;
    u8 normalizeLayerInput;

    /////////////////////////////////////////////////////////////////////////////////////
    // Behavioral state -- how is the gradient handled (learning rates, momentum, etc)
    /////////////////////////////////////////////////////////////////////////////////////

    tANN::nWeightUpRule weightUpRule;
    f64 alpha;
    f64 viscosity;     // <-- used only with the momentum weight update rule

    /////////////////////////////////////////////////////////////////////////////////////
    // Methods...
    /////////////////////////////////////////////////////////////////////////////////////

    tLayer()
    {
        // This is an invalid object. You must call init() or unpack() to setup
        // this object properly.
        layerType = tANN::kLayerTypeMax;
        weightUpRule = tANN::kWeightUpRuleMax;
    }

    void reset(f64 rmin, f64 rmax, algo::iLCG& lcg)
    {
        // Reset the node state and the gradient state.
        output.resize(0,0);
        a.resize(0,0);
        A.resize(0,0);
        da.resize(0,0);
        dA.resize(0,0);
        dw_accum.resize(0,0);
        vel.resize(0, 0);
        dw_accum_avg.resize(0, 0);

        // Randomize the weights.
        assert(rmin < rmax);
        for (i32 c = 0; c < w.cols(); c++)
        {
            for (i32 r = 0; r < w.rows(); r++)
            {
                u64 ra = lcg.next();
                w(r,c) = ((fml)ra) / ((fml)lcg.randMax());    // [0.0, 1.0]
                w(r,c) *= rmax-rmin;                          // [0.0, rmax-rmin]
                w(r,c) += rmin;                               // [rmin, rmax]
            }
        }
    }

    void init_data(i32 prevSize, i32 mySize)
    {
        // Setup the weight matrix for this layer.
        assert(prevSize > 0);
        assert(mySize > 0);
        w = Mat::Zero(mySize, prevSize+1);

        // Setup behavioral state to the default values.
        layerType = tANN::kLayerTypeHyperbolic;
        normalizeLayerInput = 1;
        weightUpRule = tANN::kWeightUpRuleNone;
        alpha = 0.0;
        viscosity = 0.0;
    }

    void init(i32 prevSize, i32 mySize,
              f64 rmin, f64 rmax, algo::iLCG& lcg)
    {
        // Init the data stuff.
        init_data(prevSize, mySize);

        // Reset the node state and randomize the weights.
        reset(rmin, rmax, lcg);
    }

    void takeInput(const Mat& input)
    {
        assert(input.rows() == w.cols());
        assert(input.rows() > 0);
        assert(input.cols() > 0);
        assert(w.rows() > 0);

        if (normalizeLayerInput)
            A.noalias() = (w * input) / (fml)input.rows();
        else
            A.noalias() = (w * input);

        if (layerType == tANN::kLayerTypeSoftmax)
        {
            tExpFunc func;
            a = A.unaryExpr(func);
            for (i32 c = 0; c < a.cols(); c++)
            {
                fml denom = a.col(c).sum();
                if (denom > 0.0)
                    a.col(c) /= denom;
                else
                    a.col(c).setConstant(1.0 / (fml)a.rows());
            }
        }
        else
        {
            tSquashFunc func(layerType);
            a = A.unaryExpr(func);
        }

        output.resize(a.rows()+1, a.cols());
        output.block(0,0,a.rows(),a.cols()) = a;
        output.bottomRows(1).setOnes();
    }

    void accumError(const Mat& input)
    {
        assert(input.rows() == w.cols());
        assert(input.rows() > 0);
        assert(input.cols() > 0);
        assert(w.rows() > 0);
        assert(input.cols() == da.cols());
        assert(da.rows() == w.rows());
        assert(A.rows() == da.rows());
        assert(A.cols() == da.cols());

        if (layerType == tANN::kLayerTypeSoftmax)
        {
            dA = da;
        }
        else
        {
            tDirSquashFunc func(layerType);
            dA.noalias() = (da.array() * A.unaryExpr(func).array()).matrix();
        }

        if (normalizeLayerInput)
            dw_accum.noalias() = (dA * input.transpose()) / (fml)input.rows();
        else
            dw_accum.noalias() = (dA * input.transpose());
    }

    void backpropagate(Mat& prev_da, const Mat& input)
    {
        assert(input.rows() == w.cols());
        assert(input.rows() > 0);
        assert(input.cols() > 0);
        assert(w.rows() > 0);
        assert(input.cols() == da.cols());
        assert(da.rows() == w.rows());
        assert(A.rows() == da.rows());
        assert(A.cols() == da.cols());
        assert(input.cols() == dA.cols());
        assert(dA.rows() == w.rows());
        assert(A.rows() == dA.rows());
        assert(A.cols() == dA.cols());

        if (normalizeLayerInput)
            prev_da.noalias() = (w.block(0,0,w.rows(),w.cols()-1).transpose() * dA) / (fml)input.rows();
        else
            prev_da.noalias() = (w.block(0,0,w.rows(),w.cols()-1).transpose() * dA);
    }

    void updateWeights(u32 batchSizeOverride = 0)
    {
        assert(a.cols() > 0);         // <--  a.cols() is the batch size
        assert(w.rows() > 0);
        assert(w.cols() > 0);
        assert(w.rows() == dw_accum.rows());
        assert(w.cols() == dw_accum.cols());

        fml batchSize = (batchSizeOverride > 0) ? ((fml)batchSizeOverride) : ((fml)a.cols());

        switch (weightUpRule)
        {
            case tANN::kWeightUpRuleNone:
            {
                break;
            }

            case tANN::kWeightUpRuleFixedLearningRate:
            {
                if (alpha <= 0.0)
                    throw eLogicError("When using the fixed learning rate rule, alpha must be set.");
                fml mult = (10.0 / batchSize) * alpha;
                w.noalias() -= mult * dw_accum;
                break;
            }

            case tANN::kWeightUpRuleMomentum:
            {
                if (alpha <= 0.0)
                    throw eLogicError("When using the momentum update rule, alpha must be set.");
                if (viscosity <= 0.0 || viscosity >= 1.0)
                    throw eLogicError("When using the momentum update rule, viscosity must be set.");
                if (vel.rows() == 0)
                    vel = Mat::Zero(w.rows(), w.cols());
                fml mult = (10.0 / batchSize) * alpha;
                vel *= viscosity;
                vel.noalias() -= mult*dw_accum;
                w.noalias() += vel;
                break;
            }

            case tANN::kWeightUpRuleAdaptiveRates:
            {
                throw eNotImplemented("This used to be implemented in the old ANN... so look there as a reference if you want to implement it here again.");
                break;
            }

            case tANN::kWeightUpRuleRPROP:
            {
                throw eNotImplemented("This used to be implemented in the old ANN... so look there as a reference if you want to implement it here again.");
                break;
            }

            case tANN::kWeightUpRuleRMSPROP:
            {
                if (alpha <= 0.0)
                    throw eLogicError("When using the rmsprop rule, alpha must be set.");
                if (dw_accum_avg.rows() == 0)
                    dw_accum_avg = Mat::Constant(w.rows(), w.cols(), 1000.0);
                fml batchNormMult = 1.0 / batchSize;
                dw_accum *= batchNormMult;
                dw_accum_avg *= 0.9;
                dw_accum_avg.noalias() += 0.1*dw_accum.array().square().matrix();
                w.noalias() -= alpha * dw_accum.binaryExpr(dw_accum_avg, t_RMSPROP_wUpdate());
                break;
            }

            case tANN::kWeightUpRuleARMS:
            {
                throw eNotImplemented("Not sure what I want here yet...");
                break;
            }

            default:
            {
                assert(false);
                break;
            }
        }
    }

    void pack(iWritable* out) const
    {
        i32 mySize = (i32)w.rows();
        i32 prevLayerSize = (i32)w.cols()-1;
        u8 type = (u8) layerType;
        u8 rule = (u8) weightUpRule;

        rho::pack(out, mySize);
        rho::pack(out, prevLayerSize);
        rho::pack(out, type);
        rho::pack(out, normalizeLayerInput);
        rho::pack(out, rule);

        switch (weightUpRule)
        {
            case tANN::kWeightUpRuleNone:
            case tANN::kWeightUpRuleRPROP:
                break;

            case tANN::kWeightUpRuleFixedLearningRate:
            case tANN::kWeightUpRuleAdaptiveRates:
            case tANN::kWeightUpRuleRMSPROP:
            case tANN::kWeightUpRuleARMS:
                rho::pack(out, alpha);
                break;

            case tANN::kWeightUpRuleMomentum:
                rho::pack(out, alpha);
                rho::pack(out, viscosity);
                // I'm not packing vel on purpose. I'll let it be all zeros on unpack().
                break;

            default:
                assert(false);
        }

        prevLayerSize += 1;
        rho::pack(out, prevLayerSize);
        for (i32 s = 0; s < prevLayerSize; s++)
        {
            rho::pack(out, mySize);
            for (i32 i = 0; i < mySize; i++)
            {
                fml f = w(i, s);
                rho::pack(out, f);
            }
        }
    }

    void unpack(iReadable* in)
    {
        i32 mySize, prevLayerSize;
        rho::unpack(in, mySize);
        rho::unpack(in, prevLayerSize);
        if (mySize <= 0 || prevLayerSize <= 0)
            throw eRuntimeError("Invalid layer stream -- invalid sizes");
        init_data(prevLayerSize, mySize);

        u8 type;
        rho::unpack(in, type);
        layerType = (tANN::nLayerType) type;
        if (layerType < 0 || layerType >= tANN::kLayerTypeMax)
            throw eRuntimeError("Invalid layer stream -- invalid layer type");

        rho::unpack(in, normalizeLayerInput);
        if (normalizeLayerInput > 1)
            throw eRuntimeError("Invalid layer stream -- invalid normalizeLayerInput");

        u8 rule;
        rho::unpack(in, rule);
        weightUpRule = (tANN::nWeightUpRule) rule;
        if (weightUpRule < 0 || weightUpRule >= tANN::kWeightUpRuleMax)
            throw eRuntimeError("Invalid layer stream -- invalid weight update rule");

        switch (weightUpRule)
        {
            case tANN::kWeightUpRuleNone:
            case tANN::kWeightUpRuleRPROP:
                break;

            case tANN::kWeightUpRuleFixedLearningRate:
            case tANN::kWeightUpRuleAdaptiveRates:
            case tANN::kWeightUpRuleRMSPROP:
            case tANN::kWeightUpRuleARMS:
                rho::unpack(in, alpha);
                break;

            case tANN::kWeightUpRuleMomentum:
                rho::unpack(in, alpha);
                rho::unpack(in, viscosity);
                // Not unpacking vel, because it was not packed.
                break;

            default:
                assert(false);
        }

        i32 prevsizecopy, mysizecopy;
        rho::unpack(in, prevsizecopy);
        if (prevsizecopy != prevLayerSize+1)
            throw eRuntimeError("Invalid layer stream -- prev size");
        for (i32 s = 0; s < prevsizecopy; s++)
        {
            rho::unpack(in, mysizecopy);
            if (mysizecopy != mySize)
                throw eRuntimeError("Invalid layer stream -- my size");
            for (i32 i = 0; i < mysizecopy; i++)
            {
                fml f;
                rho::unpack(in, f);
                w(i, s) = f;
            }
        }
    }
};


tANN::tANN(iReadable* in)
    : m_layers(NULL),
      m_numLayers(0)
{
    this->unpack(in);
}

tANN::tANN(std::vector<u32> layerSizes,
           f64 randWeightMin,
           f64 randWeightMax)
    : m_layers(NULL),
      m_numLayers(0),
      m_randWeightMin(randWeightMin),
      m_randWeightMax(randWeightMax)
{
    if (layerSizes.size() < 2)
        throw eInvalidArgument("There must be at least an input and output layer.");
    for (size_t i = 0; i < layerSizes.size(); i++)
        if (layerSizes[i] == 0)
            throw eInvalidArgument("Every layer must have size > 0");
    if (randWeightMin >= randWeightMax)
        throw eInvalidArgument("Invalid [randWeightMin, randWeightMax] range.");

    m_numLayers = (u32) layerSizes.size()-1;
                            // we don't need a layer for the input

    m_layers = new tLayer[m_numLayers];
                            // m_layers[0] is the lowest layer
                            // (i.e. first one above the input layer)

    algo::tKnuthLCG lcg;

    for (size_t i = 1; i < layerSizes.size(); i++)
    {
        m_layers[i-1].init(layerSizes[i-1], layerSizes[i],
                           m_randWeightMin, m_randWeightMax, lcg);
    }
}

void tANN::resetWeights()
{
    algo::tKnuthLCG lcg;

    for (u32 i = 0; i < m_numLayers; i++)
    {
        m_layers[i].reset(m_randWeightMin, m_randWeightMax, lcg);
    }
}

tANN::~tANN()
{
    delete [] m_layers;
    m_layers = NULL;
    m_numLayers = 0;
}

void tANN::setLayerType(nLayerType type, u32 layerIndex)
{
    if (layerIndex >= m_numLayers)
        throw eInvalidArgument("No layer with that index.");
    if (type < 0 || type >= kLayerTypeMax)
        throw eInvalidArgument("Invalid layer type");
    if (type == kLayerTypeSoftmax && layerIndex != m_numLayers-1)
        throw eInvalidArgument("Only the top layer may be a softmax group.");
    m_layers[layerIndex].layerType = type;
}

void tANN::setLayerType(nLayerType type)
{
    if (type < 0 || type >= kLayerTypeMax)
        throw eInvalidArgument("Invalid layer type");
    if (type == kLayerTypeSoftmax && m_numLayers>1)
        throw eInvalidArgument("Only the top layer may be a softmax group.");
    for (u32 i = 0; i < m_numLayers; i++)
        setLayerType(type, i);
}

void tANN::setNormalizeLayerInput(bool on, u32 layerIndex)
{
    if (layerIndex >= m_numLayers)
        throw eInvalidArgument("No layer with that index.");
    m_layers[layerIndex].normalizeLayerInput = (on ? 1 : 0);
}

void tANN::setNormalizeLayerInput(bool on)
{
    for (u32 i = 0; i < m_numLayers; i++)
        setNormalizeLayerInput(on, i);
}

void tANN::setWeightUpRule(nWeightUpRule rule, u32 layerIndex)
{
    if (layerIndex >= m_numLayers)
        throw eInvalidArgument("No layer with that index.");
    if (rule < 0 || rule >= kWeightUpRuleMax)
        throw eInvalidArgument("Invalid weight update rule");
    m_layers[layerIndex].weightUpRule = rule;
}

void tANN::setWeightUpRule(nWeightUpRule rule)
{
    if (rule < 0 || rule >= kWeightUpRuleMax)
        throw eInvalidArgument("Invalid weight update rule");
    for (u32 i = 0; i < m_numLayers; i++)
        setWeightUpRule(rule, i);
}

void tANN::setAlpha(f64 alpha, u32 layerIndex)
{
    if (layerIndex >= m_numLayers)
        throw eInvalidArgument("No layer with that index.");
    if (alpha <= 0.0)
        throw eInvalidArgument("Alpha must be greater than zero.");
    m_layers[layerIndex].alpha = alpha;
}

void tANN::setAlpha(f64 alpha)
{
    if (alpha <= 0.0)
        throw eInvalidArgument("Alpha must be greater than zero.");
    for (u32 i = 0; i < m_numLayers; i++)
        setAlpha(alpha, i);
}

void tANN::setViscosity(f64 viscosity, u32 layerIndex)
{
    if (layerIndex >= m_numLayers)
        throw eInvalidArgument("No layer with that index.");
    if (viscosity <= 0.0 || viscosity >= 1.0)
        throw eInvalidArgument("Viscosity must be greater than zero and less than one.");
    m_layers[layerIndex].viscosity = viscosity;
}

void tANN::setViscosity(f64 viscosity)
{
    if (viscosity <= 0.0 || viscosity >= 1.0)
        throw eInvalidArgument("Viscosity must be greater than zero and less than one.");
    for (u32 i = 0; i < m_numLayers; i++)
        setViscosity(viscosity, i);
}

void tANN::addExample(const tIO& input, const tIO& target)
{
    // Validate the target vector.
    if (target.size() != getNumNeuronsInLayer(m_numLayers-1))
        throw eInvalidArgument("The target vector must be the same size as the ANN's output.");
    nLayerType type = m_layers[m_numLayers-1].layerType;
    for (size_t i = 0; i < target.size(); i++)
    {
        if (target[i] < s_squash_min(type) || target[i] > s_squash_max(type))
        {
            throw eInvalidArgument("The target vector must be in the range of the "
                                   "top layer's squashing function.");
        }
    }
    if (type == kLayerTypeSoftmax)
    {
        f64 summation = 0.0;
        for (size_t i = 0; i < target.size(); i++)
            summation += target[i];
        if (summation < 0.9999 || summation > 1.0001)
        {
            throw eInvalidArgument("For networks with a softmax top layer, the sum of the target "
                    "vector must be 1.0");
        }
    }

    // Validate the input vector.
    if ((i32)input.size()+1 != m_layers[0].w.cols())
        throw eInvalidArgument("The input vector must be the same size as the ANN's input.");

    // Accumulate this example.
    m_inputAccum.push_back(input);
    m_targetAccum.push_back(target);
}

void tANN::update()
{
    if (m_inputAccum.size() == 0)
        throw eLogicError("You must give the network some examples before you call update().");

    // Build the input matrix.
    Mat input(m_inputAccum[0].size()+1, m_inputAccum.size());
    input.bottomRows(1).setOnes();
    for (size_t c = 0; c < m_inputAccum.size(); c++)
        for (size_t r = 0; r < m_inputAccum[0].size(); r++)
            input(r, c) = m_inputAccum[c][r];
    m_inputAccum.clear();

    // Build the target matrix.
    Mat target(m_targetAccum[0].size(), m_targetAccum.size());
    for (size_t c = 0; c < m_targetAccum.size(); c++)
        for (size_t r = 0; r < m_targetAccum[0].size(); r++)
            target(r, c) = m_targetAccum[c][r];
    m_targetAccum.clear();

    // Run the input through the net.
    {
        m_layers[0].takeInput(input);
        for (u32 i = 1; i < m_numLayers; i++)
            m_layers[i].takeInput(m_layers[i-1].output);
    }

    // Backprop.
    {
        assert(m_layers[m_numLayers-1].a.rows() == target.rows());
        assert(m_layers[m_numLayers-1].a.cols() == target.cols());
        m_layers[m_numLayers-1].da = m_layers[m_numLayers-1].a - target;

        for (u32 i = m_numLayers-1; i > 0; i--)
        {
            m_layers[i].accumError(m_layers[i-1].output);
            m_layers[i].backpropagate(m_layers[i-1].da, m_layers[i-1].output);
            m_layers[i].updateWeights();
        }

        m_layers[0].accumError(input);
        m_layers[0].updateWeights();
    }
}

void tANN::evaluate(const tIO& input, tIO& output) const
{
    if ((i32)input.size()+1 != m_layers[0].w.cols())
        throw eInvalidArgument("The input vector must be the same size as the ANN's input.");

    Mat inputMat(input.size()+1, 1);
    for (size_t i = 0; i < input.size(); i++)
        inputMat(i,0) = input[i];
    inputMat(input.size(),0) = 1.0;

    m_layers[0].takeInput(inputMat);
    for (u32 i = 1; i < m_numLayers; i++)
        m_layers[i].takeInput(m_layers[i-1].output);

    const Mat& outMat = m_layers[m_numLayers-1].a;
    assert(outMat.rows() > 0);
    assert(outMat.cols() == 1);
    output.resize(outMat.rows());
    for (i32 i = 0; i < outMat.rows(); i++)
        output[i] = outMat(i,0);
}

void tANN::evaluateBatch(const std::vector<tIO>& inputs,
                               std::vector<tIO>& outputs) const
{
    outputs.resize(inputs.size());
    evaluateBatch(inputs.begin(), inputs.end(), outputs.begin());
}

void tANN::evaluateBatch(std::vector<tIO>::const_iterator inputStart,
                         std::vector<tIO>::const_iterator inputEnd,
                         std::vector<tIO>::iterator outputStart) const
{
    if (inputStart >= inputEnd)
        throw eInvalidArgument("There must be at least one input example.");

    for (std::vector<tIO>::const_iterator initr = inputStart; initr != inputEnd; initr++)
        if ((i32)((initr->size())+1) != m_layers[0].w.cols())
            throw eInvalidArgument("All input vectors must be the same size as the ANN's input.");

    // Build the input matrix.
    Mat inputMat((inputStart->size())+1, inputEnd-inputStart);
    inputMat.bottomRows(1).setOnes();
    std::vector<tIO>::const_iterator initr = inputStart;
    for (i32 c = 0; c < inputMat.cols(); c++)
    {
        for (size_t r = 0; r < initr->size(); r++)
            inputMat(r,c) = (*initr)[r];
        initr++;
    }

    // Run the input through the net.
    m_layers[0].takeInput(inputMat);
    for (u32 i = 1; i < m_numLayers; i++)
        m_layers[i].takeInput(m_layers[i-1].output);

    // Capture the output.
    const Mat& outMat = m_layers[m_numLayers-1].a;
    assert(outMat.rows() > 0);
    assert(outMat.cols() == inputMat.cols());
    std::vector<tIO>::iterator outitr = outputStart;
    for (i32 c = 0; c < outMat.cols(); c++)
    {
        outitr->resize(outMat.rows());
        for (size_t r = 0; r < outitr->size(); r++)
            (*outitr)[r] = outMat(r,c);
        outitr++;
    }
}

f64 tANN::calculateError(const tIO& output, const tIO& target)
{
    if (m_layers[m_numLayers-1].layerType == kLayerTypeSoftmax)
        return crossEntropyCost(output, target);
    else
        return standardSquaredError(output, target);
}

f64 tANN::calculateError(const std::vector<tIO>& outputs,
                         const std::vector<tIO>& targets)
{
    if (m_layers[m_numLayers-1].layerType == kLayerTypeSoftmax)
        return crossEntropyCost(outputs, targets);
    else
        return standardSquaredError(outputs, targets);
}

void tANN::reset()
{
    resetWeights();
}

void tANN::printLearnerInfo(std::ostream& out) const
{
    const int colw = 20;
    std::ostringstream sout;

    sout << "Artificial Neural Network Info:" << endl;

    // Layer type (and normalizeLayerInput):
    sout << "             layer type:";
    sout << std::right << std::setw(colw) << "input";
    for (u32 i = 0; i < m_numLayers; i++)
    {
        string print = s_layerTypeToString(m_layers[i].layerType);
        if (m_layers[i].normalizeLayerInput)
            print += "(norm'd)";
        sout << std::right << std::setw(colw) << print;
    }
    sout << endl;

    // Weight update rule:
    sout << "     weight update rule:";
    sout << std::right << std::setw(colw) << "-";
    for (u32 i = 0; i < m_numLayers; i++)
    {
        std::ostringstream ss;
        ss << s_weightUpRuleToString(m_layers[i].weightUpRule);
        switch (m_layers[i].weightUpRule)
        {
            case kWeightUpRuleNone:
            case kWeightUpRuleRPROP:
                break;
            case kWeightUpRuleFixedLearningRate:
            case kWeightUpRuleAdaptiveRates:
            case kWeightUpRuleRMSPROP:
            case kWeightUpRuleARMS:
                ss << "(a=" << m_layers[i].alpha << ")";
                break;
            case kWeightUpRuleMomentum:
                ss << "(a=" << m_layers[i].alpha
                   << ",v=" << m_layers[i].viscosity << ")";
                break;
            default:
                assert(false);
        }
        sout << std::right << std::setw(colw) << ss.str();
    }
    sout << endl;

    // Network size:
    sout << "             layer size:";
    sout << std::right << std::setw(colw) << m_layers[0].w.cols()-1;
    for (u32 i = 0; i < m_numLayers; i++)
        sout << std::right << std::setw(colw) << m_layers[i].w.rows();
    sout << endl;

    // Num free parameters:
    sout << "    num free parameters:";
    sout << std::right << std::setw(colw) << "-";
    for (u32 i = 0; i < m_numLayers; i++)
        sout << std::right << std::setw(colw) << (m_layers[i].w.rows()*m_layers[i].w.cols());
    sout << endl;

    sout << endl;

    out << sout.str() << std::flush;
}

string tANN::learnerInfoString() const
{
    std::ostringstream out;

    out << "size=" << m_layers[0].w.cols()-1;
    for (u32 i = 0; i < m_numLayers; i++)
        out << '-' << m_layers[i].w.rows();

    out << "__type=" << 'i';
    for (u32 i = 0; i < m_numLayers; i++)
    {
        out << '-' << s_layerTypeToChar(m_layers[i].layerType);
        if (m_layers[i].normalizeLayerInput)
            out << 'n';
    }

    out << "__wrule=" << 'i';
    for (u32 i = 0; i < m_numLayers; i++)
    {
        out << '-' << s_weightUpRuleToChar(m_layers[i].weightUpRule);
        switch (m_layers[i].weightUpRule)
        {
            case kWeightUpRuleNone:
            case kWeightUpRuleRPROP:
                break;
            case kWeightUpRuleFixedLearningRate:
            case kWeightUpRuleAdaptiveRates:
            case kWeightUpRuleRMSPROP:
            case kWeightUpRuleARMS:
                out << m_layers[i].alpha;
                break;
            case kWeightUpRuleMomentum:
                out << m_layers[i].alpha << ',' << m_layers[i].viscosity;
                break;
            default:
                assert(false);
        }
    }

    return out.str();
}

u32 tANN::getNumLayers() const
{
    return m_numLayers;
}

u32 tANN::getNumNeuronsInLayer(u32 layerIndex) const
{
    if (layerIndex >= m_numLayers)
        throw eInvalidArgument("No layer with that index.");
    return (u32) m_layers[layerIndex].w.rows();
}

void tANN::getWeights(u32 layerIndex, u32 neuronIndex, tIO& weights) const
{
    if (neuronIndex >= getNumNeuronsInLayer(layerIndex))
        throw eInvalidArgument("No layer/node with that index.");

    const Mat& w = m_layers[layerIndex].w;
    assert(w.cols() > 1);
    weights.resize(w.cols()-1);
    for (size_t s = 0; s < weights.size(); s++)
        weights[s] = w(neuronIndex,s);
}

f64 tANN::getBias(u32 layerIndex, u32 neuronIndex) const
{
    if (neuronIndex >= getNumNeuronsInLayer(layerIndex))
        throw eInvalidArgument("No layer/node with that index.");
    const Mat& w = m_layers[layerIndex].w;
    assert(w.cols() > 1);
    return w.rightCols(1)(neuronIndex,0);
}

f64 tANN::getOutput(u32 layerIndex, u32 neuronIndex) const
{
    if (neuronIndex >= getNumNeuronsInLayer(layerIndex))
        throw eInvalidArgument("No layer/node with that index.");
    const Mat& a = m_layers[layerIndex].a;
    if (a.cols() == 0)
        throw eInvalidArgument("There is no \"most recent\" output of this neuron.");
    return a(neuronIndex,a.cols()-1);
}

void tANN::getImage(u32 layerIndex, u32 neuronIndex,
              bool color, u32 width, bool absolute,
              img::tImage* dest) const
{
    // Get the weights.
    tIO weights;
    getWeights(layerIndex, neuronIndex, weights);
    assert(weights.size() > 0);

    // Use the image creating method in ml::common to do the work.
    un_examplify(weights, color, width, absolute, dest);
    u32 height = dest->height();

    // Add an output indicator.
    nLayerType type = m_layers[layerIndex].layerType;
    f64 output = (getOutput(layerIndex, neuronIndex) - s_squash_min(type))
                    / (s_squash_max(type) - s_squash_min(type));   // output now in [0, 1]
    u8 outputByte = (u8) (output*255.0);
    u8 red = 0;
    u8 green = (u8) (255 - outputByte);
    u8 blue = outputByte;
    u32 ySpan = height / 5;
    u32 yStart = 0;
    u32 xSpan = width / 5;
    u32 xStart = (u32) (output * (width-xSpan));
    for (u32 r = yStart; r < yStart+ySpan; r++)
    {
        for (u32 c = xStart; c < xStart+xSpan; c++)
        {
            u8* buf = dest->buf() + r*dest->width()*3 + c*3;
            buf[0] = red;
            buf[1] = green;
            buf[2] = blue;
        }
    }
}

void tANN::pack(iWritable* out) const
{
    rho::pack(out, m_numLayers);
    for (u32 i = 0; i < m_numLayers; i++)
        m_layers[i].pack(out);
    rho::pack(out, m_randWeightMin);
    rho::pack(out, m_randWeightMax);
}

void tANN::unpack(iReadable* in)
{
    // Try to unpack the network.
    u32 numLayers;
    f64 randWeightMin, randWeightMax;
    rho::unpack(in, numLayers);
    if (numLayers == 0)
        throw eRuntimeError("Invalid ANN stream -- num layers");
    tLayer* layers = new tLayer[numLayers];
    try
    {
        for (u32 i = 0; i < numLayers; i++)
            layers[i].unpack(in);
        rho::unpack(in, randWeightMin);
        rho::unpack(in, randWeightMax);
        if (randWeightMin >= randWeightMax)
            throw eRuntimeError("Invalid ANN stream -- rand range");
    }
    catch (ebObject& e)
    {
        delete [] layers;
        throw;
    }

    // If it worked, clobber the current network.
    m_numLayers = numLayers;
    delete [] m_layers;
    m_layers = layers;
    m_randWeightMin = randWeightMin;
    m_randWeightMax = randWeightMax;
    m_inputAccum.clear();
    m_targetAccum.clear();
}


}   // namespace ml


#include "tCNN.ipp"    // this is done because tCNN uses tLayer, and this will
                       // allow the compiler to optimize the code a lot
