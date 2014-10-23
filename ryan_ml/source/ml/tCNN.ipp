#include <ml/tCNN.h>

#include <rho/algo/string_util.h>


namespace ml
{


class tLayerCNN : public bNonCopyable
{
  private:

    u32 m_inputSize;     // Information about this CNN-layer's input.
    u32 m_inputWidth;    // ...
    u32 m_inputHeight;   // ...

    u32 m_receptiveFieldWidth;    // The size of this CNN-layer's
    u32 m_receptiveFieldHeight;   // receptive field.

    u32 m_stepSizeX;   // The step size for the receptive field.
    u32 m_stepSizeY;   // ...

    u32 m_stepsX;   // The number of times you can step the
    u32 m_stepsY;   // receptive field.

    u32 m_poolWidth;    // The output max-pooling size.
    u32 m_poolHeight;   // ...

  private:

    u32 m_numFeatureMapsInThisLayer;   // The number of features maps that this
                                       // CNN-layer will have. That is,
                                       // the number of unique filters that
                                       // will be trained then replicated
                                       // across the receptive fields.

    tLayer m_layer;          // The layer that contains the trainable filters.
                             // There are m_numFeatureMapsInThisLayer number of
                             // neurons in this layer.

    u32 m_numReplicas;          // The number of times the layer above will be replicated
                                // across the receptive fields.
                                // Equals ( (m_stepsX+1) * (m_stepsY+1) )

  private:

    // The following is state to make calculations faster so that you don't
    // have to reallocate vectors all the time.

    Mat m_output;      // The un-convolved output of this layer.
    Mat m_da;          // The un-convolved error coming into this layer.

    Mat m_pooledOutput;  // Only used when (m_poolWidth > 1 || m_poolHeight > 1)
    Mat m_pooled_da;     // (same)

    Mat m_convolvedInput;   // The convolved input to this layer.
    Mat m_convolved_da;     // The convolved error coming into this layer.

  private:

    void m_fillField(const Mat& input, i32 c, u32 x, u32 y, Mat& fieldInput, i32 f) const
    {
        u32 inputIndex = y*m_inputWidth + x;
        u32 fieldInputIndex = 0;
        for (u32 yy = 0; yy < m_receptiveFieldHeight; yy++)
        {
            for (u32 xx = 0; xx < m_receptiveFieldWidth; xx++)
            {
                fieldInput(fieldInputIndex++,f) = input(inputIndex+xx,c);
            }
            inputIndex += m_inputWidth;
        }
    }

    void m_reverseFillField(const Mat& fieldInput, i32 f, u32 x, u32 y, Mat& input, i32 c) const
    {
        u32 inputIndex = y*m_inputWidth + x;
        u32 fieldInputIndex = 0;
        for (u32 yy = 0; yy < m_receptiveFieldHeight; yy++)
        {
            for (u32 xx = 0; xx < m_receptiveFieldWidth; xx++)
            {
                 input(inputIndex+xx,c) += fieldInput(fieldInputIndex++,f);
            }
            inputIndex += m_inputWidth;
        }
    }

    void m_poolOutput()
    {
        assert(m_output.rows() == m_numFeatureMapsInThisLayer*m_numReplicas);
        assert(m_output.cols() > 0);
        assert(m_pooledOutput.rows() == ((m_stepsX+1)/m_poolWidth) * ((m_stepsY+1)/m_poolHeight) * m_numFeatureMapsInThisLayer);

        size_t outWidth = (m_stepsX+1) * m_numFeatureMapsInThisLayer;
        m_pooledOutput.resize(m_pooledOutput.rows(), m_output.cols());
        for (i32 c = 0; c < m_output.cols(); c++)
        {
            size_t pooledoutIndex = 0;
            for (u32 y = 0; y <= (m_stepsY+1-m_poolHeight); y += m_poolHeight)
            {
                for (u32 x = 0; x <= (m_stepsX+1-m_poolWidth); x += m_poolWidth)
                {
                    size_t outputIndex = y*outWidth + x*m_numFeatureMapsInThisLayer;
                    for (u32 m = 0; m < m_numFeatureMapsInThisLayer; m++)
                    {
                        size_t off = outputIndex + m;
                        f64 maxval = m_output(off,c);
                        for (u32 i = 0; i < m_poolHeight; i++)
                        {
                            size_t off2 = off;
                            for (u32 j = 0; j < m_poolWidth; j++)
                            {
                                maxval = std::max(maxval, m_output(off2,c));
                                off2 += m_numFeatureMapsInThisLayer;
                            }
                            off += outWidth;
                        }
                        m_pooledOutput(pooledoutIndex++,c) = maxval;
                    }
                }
            }
            assert(pooledoutIndex == (u32)m_pooledOutput.rows());
        }
    }

    void m_unpool_da_sparse()
    {
        assert(m_output.rows() == m_numFeatureMapsInThisLayer*m_numReplicas);
        assert(m_output.cols() > 0);
        assert(m_output.rows() == m_da.rows());
        assert(m_pooled_da.rows() == ((m_stepsX+1)/m_poolWidth) * ((m_stepsY+1)/m_poolHeight) * m_numFeatureMapsInThisLayer);
        assert(m_pooled_da.cols() == m_output.cols());
        assert(m_pooled_da.rows() == m_pooledOutput.rows());
        assert(m_pooled_da.cols() == m_pooledOutput.cols());

        size_t outWidth = (m_stepsX+1) * m_numFeatureMapsInThisLayer;
        m_da.resize(m_da.rows(), m_output.cols());
        m_da.setZero();
        for (i32 c = 0; c < m_da.cols(); c++)
        {
            size_t pooledoutIndex = 0;
            for (u32 y = 0; y <= (m_stepsY+1-m_poolHeight); y += m_poolHeight)
            {
                for (u32 x = 0; x <= (m_stepsX+1-m_poolWidth); x += m_poolWidth)
                {
                    size_t outputIndex = y*outWidth + x*m_numFeatureMapsInThisLayer;
                    for (u32 m = 0; m < m_numFeatureMapsInThisLayer; m++)
                    {
                        size_t off = outputIndex + m;
                        f64 maxval = m_output(off,c);
                        size_t maxi = off;
                        for (u32 i = 0; i < m_poolHeight; i++)
                        {
                            size_t off2 = off;
                            for (u32 j = 0; j < m_poolWidth; j++)
                            {
                                if (maxval < m_output(off2,c))
                                {
                                    maxval = m_output(off2,c);
                                    maxi = off2;
                                }
                                off2 += m_numFeatureMapsInThisLayer;
                            }
                            off += outWidth;
                        }
                        m_da(maxi,c) = m_pooled_da(pooledoutIndex++,c);
                    }
                }
            }
            assert(pooledoutIndex == (u32)m_pooled_da.rows());
        }
    }

  public:

    tLayerCNN()
    {
        // This constructed object is invalid. You must call init() or unpack()
        // to set it up properly.
    }

    void init_data()
    {
        m_output = Mat::Zero((m_stepsX+1)*(m_stepsY+1) * m_numFeatureMapsInThisLayer, 0);
        m_da = m_output;
        m_pooledOutput = Mat::Zero(((m_stepsX+1)/m_poolWidth) * ((m_stepsY+1)/m_poolHeight) * m_numFeatureMapsInThisLayer, 0);
        m_pooled_da = m_pooledOutput;
    }

    void init(u32 inputSize, u32 inputRowWidth,
              u32 receptiveFieldWidth, u32 receptiveFieldHeight,
              u32 stepSizeHorizontal, u32 stepSizeVertical,
              u32 numFeatureMapsInThisLayer,
              u32 poolWidth, u32 poolHeight,
              f64 rmin, f64 rmax, algo::iLCG& lcg)
    {
        // Validation and setup of useful member state.
        if (!(inputSize > 0))
            throw eInvalidArgument("Assert failed: (inputSize > 0)");
        if (!(inputRowWidth > 0))
            throw eInvalidArgument("Assert failed: (inputRowWidth > 0)");
        if (!((inputSize % inputRowWidth) == 0))
            throw eInvalidArgument("Assert failed: ((inputSize % inputRowWidth) == 0)");
        m_inputSize = inputSize;
        m_inputWidth = inputRowWidth;
        m_inputHeight = (inputSize / inputRowWidth);

        if (!(receptiveFieldWidth > 0))
            throw eInvalidArgument("Assert failed: (receptiveFieldWidth > 0)");
        if (!(receptiveFieldHeight > 0))
            throw eInvalidArgument("Assert failed: (receptiveFieldHeight > 0)");
        if (!(m_inputWidth >= receptiveFieldWidth))
            throw eInvalidArgument("Assert failed: (m_inputWidth >= receptiveFieldWidth)");
        if (!(m_inputHeight >= receptiveFieldHeight))
            throw eInvalidArgument("Assert failed: (m_inputHeight >= receptiveFieldHeight)");
        m_receptiveFieldWidth = receptiveFieldWidth;
        m_receptiveFieldHeight = receptiveFieldHeight;
        u32 numPossibleStepsX = m_inputWidth - m_receptiveFieldWidth;
        u32 numPossibleStepsY = m_inputHeight - m_receptiveFieldHeight;

        if (!(stepSizeHorizontal > 0))
            throw eInvalidArgument("Assert failed: (stepSizeHorizontal > 0)");
        if (!(stepSizeVertical > 0))
            throw eInvalidArgument("Assert failed: (stepSizeVertical > 0)");
        m_stepSizeX = stepSizeHorizontal;
        m_stepSizeY = stepSizeVertical;
        m_stepsX = numPossibleStepsX / m_stepSizeX;
        m_stepsY = numPossibleStepsY / m_stepSizeY;

        if (!(numFeatureMapsInThisLayer > 0))
            throw eInvalidArgument("Assert failed: (numFeatureMapsInThisLayer > 0)");
        m_numFeatureMapsInThisLayer = numFeatureMapsInThisLayer;

        if (!(poolWidth > 0))
            throw eInvalidArgument("Assert failed: (poolWidth > 0)");
        if (!(poolHeight > 0))
            throw eInvalidArgument("Assert failed: (poolHeight > 0)");
        if (!(poolWidth <= (m_stepsX+1)))
            throw eInvalidArgument("Assert failed: (poolWidth <= (m_stepsX+1))");
        if (!(poolHeight <= (m_stepsY+1)))
            throw eInvalidArgument("Assert failed: (poolHeight <= (m_stepsY+1))");
        m_poolWidth = poolWidth;
        m_poolHeight = poolHeight;

        assert(rmin < rmax);

        m_numReplicas = (m_stepsX+1)*(m_stepsY+1);
        m_layer.init(m_receptiveFieldWidth * m_receptiveFieldHeight,
                     m_numFeatureMapsInThisLayer,
                     rmin, rmax, lcg);

        init_data();
    }

    void resetWeights(f64 rmin, f64 rmax, algo::iLCG& lcg)
    {
        assert(rmin < rmax);
        m_layer.reset(rmin, rmax, lcg);
    }

    void setLayerType(tANN::nLayerType type)
    {
        assert(type >= 0 && type < tANN::kLayerTypeMax);
        m_layer.layerType = type;
    }

    void setLayerNormalizeLayerInput(bool norm)
    {
        m_layer.normalizeLayerInput = (norm ? 1 : 0);
    }

    void setLayerWeightUpdateRule(tANN::nWeightUpRule rule)
    {
        assert(rule >= 0 && rule < tANN::kWeightUpRuleMax);
        m_layer.weightUpRule = rule;
    }

    void setLayerAlpha(f64 alpha)
    {
        assert(alpha > 0.0);
        m_layer.alpha = alpha;
    }

    void setLayerViscosity(f64 viscosity)
    {
        assert(viscosity > 0.0 && viscosity < 1.0);
        m_layer.viscosity = viscosity;
    }

    const tLayer& getLayer() const
    {
        return m_layer;
    }

    u32 getNumReplicas() const
    {
        return m_numReplicas;
    }

    u32 getInputSize() const
    {
        return m_inputSize;
    }

    u32 getInputWidth() const
    {
        return m_inputWidth;
    }

    u32 getInputHeight() const
    {
        return m_inputHeight;
    }

    u32 getReceptiveFieldWidth() const
    {
        return m_receptiveFieldWidth;
    }

    u32 getReceptiveFieldHeight() const
    {
        return m_receptiveFieldHeight;
    }

    u32 getStepSizeX() const
    {
        return m_stepSizeX;
    }

    u32 getStepSizeY() const
    {
        return m_stepSizeY;
    }

    u32 getStepsX() const
    {
        return m_stepsX;
    }

    u32 getStepsY() const
    {
        return m_stepsY;
    }

    u32 getNumFeatureMaps() const
    {
        return m_numFeatureMapsInThisLayer;
    }

    u32 getPoolWidth() const
    {
        return m_poolWidth;
    }

    u32 getPoolHeight() const
    {
        return m_poolHeight;
    }

    u32 getNumFreeParameters() const
    {
        return (u32)(m_layer.w.cols() * m_layer.w.rows());
    }

    void takeInput(const Mat& input)
    {
        assert(input.rows() == m_inputSize);
        assert(input.cols() > 0);

        // Convolve the input.
        m_convolvedInput.resize(m_receptiveFieldWidth*m_receptiveFieldHeight+1,
                                input.cols() * m_numReplicas);
        m_convolvedInput.bottomRows(1).setOnes();
        u32 convolvCol = 0;
        for (i32 c = 0; c < input.cols(); c++)
        {
            for (u32 y = 0; y < m_inputHeight-m_receptiveFieldHeight+1; y += m_stepSizeY)
            {
                for (u32 x = 0; x < m_inputWidth-m_receptiveFieldWidth+1; x += m_stepSizeX)
                {
                    m_fillField(input, c, x, y, m_convolvedInput, convolvCol++);
                }
            }
        }
        assert(convolvCol == (m_stepsX+1)*(m_stepsY+1)*input.cols());

        // Run the convolved input through the ANN layer.
        m_layer.takeInput(m_convolvedInput);

        // Put the output of each filter into one output vector.
        assert(m_output.rows() == m_numFeatureMapsInThisLayer*m_numReplicas);
        assert(m_layer.a.rows() == m_numFeatureMapsInThisLayer);
        assert(m_layer.a.cols() == input.cols()*m_numReplicas);
        m_output.resize(m_output.rows(), input.cols());
        convolvCol = 0;
        for (i32 c = 0; c < input.cols(); c++)
        {
            size_t outputIndex = 0;
            for (u32 i = 0; i < m_numReplicas; i++)
            {
                for (u32 j = 0; j < m_layer.a.rows(); j++)
                    m_output(outputIndex++,c) = m_layer.a(j,convolvCol);
                convolvCol++;
            }
        }

        // Pool the output vector (only if we do pooling in this layer)
        if (m_poolWidth > 1 || m_poolHeight > 1)
        {
            m_poolOutput();   // <-- fills m_pooledOutput from the contents of m_output
        }
    }

    const Mat& getOutput()
    {
        if (m_poolWidth > 1 || m_poolHeight > 1)
            return m_pooledOutput;
        else
            return m_output;
    }

    const Mat& getRealOutput()
    {
        return m_output;
    }

    Mat& get_da()
    {
        if (m_poolWidth > 1 || m_poolHeight > 1)
            return m_pooled_da;
        else
            return m_da;
    }

    void distribute_da(const Mat& input)
    {
        // If we do max pooling, we will need to expand the pooled da.
        if (m_poolWidth > 1 || m_poolHeight > 1)
        {
            m_unpool_da_sparse();  // <-- fills m_da from the contents of m_pooled_da
        }

        assert(m_da.rows() == m_numFeatureMapsInThisLayer*m_numReplicas);
        assert(m_da.cols() == input.cols());

        assert(m_layer.a.rows() == m_numFeatureMapsInThisLayer);
        assert(m_layer.a.cols() == input.cols()*m_numReplicas);

        // Distribute.
        m_layer.da.resize(m_layer.a.rows(), m_layer.a.cols());
        u32 convolvCol = 0;
        for (i32 c = 0; c < input.cols(); c++)
        {
            size_t outputIndex = 0;
            for (u32 i = 0; i < m_numReplicas; i++)
            {
                for (u32 j = 0; j < m_layer.da.rows(); j++)
                    m_layer.da(j,convolvCol) = m_da(outputIndex++,c);
                convolvCol++;
            }
        }
    }

    void accumError()
    {
        m_layer.accumError(m_convolvedInput);
    }

    void backpropagate(Mat& prev_da, const Mat& input)
    {
        // Back-propagate the convolved stuff.
        m_layer.backpropagate(m_convolved_da, m_convolvedInput);
        assert(m_convolved_da.rows()+1 == m_convolvedInput.rows());
        assert(m_convolved_da.cols() == m_convolvedInput.cols());
        assert(m_convolved_da.rows() == m_receptiveFieldWidth*m_receptiveFieldHeight);
        assert(m_convolved_da.cols() == input.cols()*m_numReplicas);
        assert(input.rows() == m_inputSize);
        assert(input.cols() > 0);

        // Un-convolve the convolved da.
        prev_da.resize(input.rows(), input.cols());
        prev_da.setZero();
        u32 convolvCol = 0;
        for (i32 c = 0; c < prev_da.cols(); c++)
        {
            for (u32 y = 0; y < m_inputHeight-m_receptiveFieldHeight+1; y += m_stepSizeY)
            {
                for (u32 x = 0; x < m_inputWidth-m_receptiveFieldWidth+1; x += m_stepSizeX)
                {
                    m_reverseFillField(m_convolved_da, convolvCol++, x, y, prev_da, c);
                }
            }
        }
        assert(convolvCol == (m_stepsX+1)*(m_stepsY+1)*prev_da.cols());
    }

    void updateWeights(const Mat& input)
    {
        m_layer.updateWeights((u32)input.cols());
    }

    void pack(iWritable* out) const
    {
        rho::pack(out, m_inputSize);
        rho::pack(out, m_inputWidth);
        rho::pack(out, m_inputHeight);
        rho::pack(out, m_receptiveFieldWidth);
        rho::pack(out, m_receptiveFieldHeight);
        rho::pack(out, m_stepSizeX);
        rho::pack(out, m_stepSizeY);
        rho::pack(out, m_stepsX);
        rho::pack(out, m_stepsY);
        rho::pack(out, m_poolWidth);
        rho::pack(out, m_poolHeight);
        rho::pack(out, m_numFeatureMapsInThisLayer);
        rho::pack(out, m_numReplicas);

        m_layer.pack(out);
    }

    void unpack(iReadable* in)
    {
        rho::unpack(in, m_inputSize);
        rho::unpack(in, m_inputWidth);
        rho::unpack(in, m_inputHeight);
        rho::unpack(in, m_receptiveFieldWidth);
        rho::unpack(in, m_receptiveFieldHeight);
        rho::unpack(in, m_stepSizeX);
        rho::unpack(in, m_stepSizeY);
        rho::unpack(in, m_stepsX);
        rho::unpack(in, m_stepsY);
        rho::unpack(in, m_poolWidth);
        rho::unpack(in, m_poolHeight);
        rho::unpack(in, m_numFeatureMapsInThisLayer);
        rho::unpack(in, m_numReplicas);

        if (m_numReplicas == 0)
            throw eRuntimeError("Invalid CNN stream -- num replicas");

        m_layer.unpack(in);

        init_data();
    }
};


static
u32 s_toInt(const string& str)
{
    std::istringstream in(str);
    u32 val;
    if (!(in >> val))
        throw eInvalidArgument("Expected numeric value where there was not one.");
    return val;
}


tCNN::tCNN(string descriptionString)
    : m_layers(NULL),
      m_numLayers(0)
{
    m_randWeightMin = -1.0;
    m_randWeightMax = 1.0;
    algo::tKnuthLCG lcg;

    if (m_randWeightMin >= m_randWeightMax)
        throw eInvalidArgument("Invalid initial rand weight range.");

    {
        m_numLayers = 0;
        std::istringstream in(descriptionString);
        string line;
        getline(in, line);
        while (getline(in, line)) m_numLayers++;
    }

    if (m_numLayers == 0)
        throw eInvalidArgument("There must be at least one layer in the CNN.");

    m_layers = new tLayerCNN[m_numLayers];
                            // m_layers[0] is the lowest layer
                            // (i.e. first one above the input layer)

    try
    {
        std::istringstream in(descriptionString);
        string line;

        u32 width, height;
        {
            if (!getline(in, line))
                throw eInvalidArgument("Not sure why this failed.");
            std::istringstream linein(line);
            if (!(linein >> width >> height))
                throw eInvalidArgument("Cannot read input description line.");
        }

        u32 nmaps = 1;

        for (u32 i = 0; i < m_numLayers; i++)
        {
            if (!getline(in, line))
                throw eInvalidArgument("Not sure why this failed.");

            std::vector<string> parts = algo::split(line, " ");

            if (parts.size() == 7)
            {
                // A convolutional layer:

                u32 rfwidth = s_toInt(parts[0]);
                u32 rfheight = s_toInt(parts[1]);
                u32 rfstepx = s_toInt(parts[2]);
                u32 rfstepy = s_toInt(parts[3]);
                u32 nmapsHere = s_toInt(parts[4]);
                u32 poolwidth = s_toInt(parts[5]);
                u32 poolheight = s_toInt(parts[6]);

                m_layers[i].init(width*height*nmaps,  //   u32 inputSize
                                 width*nmaps,         //   u32 inputRowWidth
                                 rfwidth*nmaps,       //   u32 receptiveFieldWidth
                                 rfheight,            //   u32 receptiveFieldHeight
                                 rfstepx*nmaps,       //   u32 stepSizeHorizontal
                                 rfstepy,             //   u32 stepSizeVertical
                                 nmapsHere,           //   u32 numFeatureMapsInThisLayer
                                 poolwidth,           //   u32 poolWidth
                                 poolheight,          //   u32 poolHeight
                                 m_randWeightMin,     //   f64 rmin
                                 m_randWeightMax,     //   f64 rmax
                                 lcg);                //   algo::iLCG& lcg

                nmaps = nmapsHere;
                width = (m_layers[i].getStepsX()+1) / poolwidth;
                height = (m_layers[i].getStepsY()+1) / poolheight;
            }
            else if (parts.size() == 1)
            {
                // A fully-connected layer:

                u32 numouts = s_toInt(parts[0]);

                m_layers[i].init(width*height*nmaps,  //   u32 inputSize
                                 width*nmaps,         //   u32 inputRowWidth
                                 width*nmaps,         //   u32 receptiveFieldWidth
                                 height,              //   u32 receptiveFieldHeight
                                 1,                   //   u32 stepSizeHorizontal
                                 1,                   //   u32 stepSizeVertical
                                 numouts,             //   u32 numFeatureMapsInThisLayer
                                 1,                   //   u32 poolWidth
                                 1,                   //   u32 poolHeight
                                 m_randWeightMin,     //   f64 rmin
                                 m_randWeightMax,     //   f64 rmax
                                 lcg);                //   algo::iLCG& lcg

                nmaps = numouts;
                width = 1;
                height = 1;
            }
            else
            {
                throw eInvalidArgument("Invalid line in description string.");
            }
        }
    }
    catch (std::exception& e)
    {
        delete [] m_layers;
        m_layers = NULL;
        m_numLayers = 0;
        throw;
    }
}

tCNN::tCNN(iReadable* in)
    : m_layers(NULL),
      m_numLayers(0)
{
    this->unpack(in);
}

void tCNN::resetWeights()
{
    algo::tKnuthLCG lcg;

    for (u32 i = 0; i < m_numLayers; i++)
    {
        m_layers[i].resetWeights(m_randWeightMin, m_randWeightMax, lcg);
    }
}

tCNN::~tCNN()
{
    delete [] m_layers;
    m_layers = NULL;
    m_numLayers = 0;
}

void tCNN::setLayerType(tANN::nLayerType type, u32 layerIndex)
{
    if (layerIndex >= m_numLayers)
        throw eInvalidArgument("No layer with that index.");
    if (type < 0 || type >= tANN::kLayerTypeMax)
        throw eInvalidArgument("Invalid layer type");
    if (type == tANN::kLayerTypeSoftmax && layerIndex != m_numLayers-1)
        throw eInvalidArgument("Only the top layer may be a softmax group.");
    m_layers[layerIndex].setLayerType(type);
}

void tCNN::setLayerType(tANN::nLayerType type)
{
    if (type < 0 || type >= tANN::kLayerTypeMax)
        throw eInvalidArgument("Invalid layer type");
    if (type == tANN::kLayerTypeSoftmax && m_numLayers>1)
        throw eInvalidArgument("Only the top layer may be a softmax group.");
    for (u32 i = 0; i < m_numLayers; i++)
        setLayerType(type, i);
}

void tCNN::setNormalizeLayerInput(bool on, u32 layerIndex)
{
    if (layerIndex >= m_numLayers)
        throw eInvalidArgument("No layer with that index.");
    m_layers[layerIndex].setLayerNormalizeLayerInput(on);
}

void tCNN::setNormalizeLayerInput(bool on)
{
    for (u32 i = 0; i < m_numLayers; i++)
        setNormalizeLayerInput(on, i);
}

void tCNN::setWeightUpRule(tANN::nWeightUpRule rule, u32 layerIndex)
{
    if (layerIndex >= m_numLayers)
        throw eInvalidArgument("No layer with that index.");
    if (rule < 0 || rule >= tANN::kWeightUpRuleMax)
        throw eInvalidArgument("Invalid weight update rule");
    m_layers[layerIndex].setLayerWeightUpdateRule(rule);
}

void tCNN::setWeightUpRule(tANN::nWeightUpRule rule)
{
    if (rule < 0 || rule >= tANN::kWeightUpRuleMax)
        throw eInvalidArgument("Invalid weight update rule");
    for (u32 i = 0; i < m_numLayers; i++)
        setWeightUpRule(rule, i);
}

void tCNN::setAlpha(f64 alpha, u32 layerIndex)
{
    if (layerIndex >= m_numLayers)
        throw eInvalidArgument("No layer with that index.");
    if (alpha <= 0.0)
        throw eInvalidArgument("Alpha must be greater than zero.");
    m_layers[layerIndex].setLayerAlpha(alpha);
}

void tCNN::setAlpha(f64 alpha)
{
    if (alpha <= 0.0)
        throw eInvalidArgument("Alpha must be greater than zero.");
    for (u32 i = 0; i < m_numLayers; i++)
        setAlpha(alpha, i);
}

void tCNN::setViscosity(f64 viscosity, u32 layerIndex)
{
    if (layerIndex >= m_numLayers)
        throw eInvalidArgument("No layer with that index.");
    if (viscosity <= 0.0 || viscosity >= 1.0)
        throw eInvalidArgument("Viscosity must be greater than zero and less than one.");
    m_layers[layerIndex].setLayerViscosity(viscosity);
}

void tCNN::setViscosity(f64 viscosity)
{
    if (viscosity <= 0.0 || viscosity >= 1.0)
        throw eInvalidArgument("Viscosity must be greater than zero and less than one.");
    for (u32 i = 0; i < m_numLayers; i++)
        setViscosity(viscosity, i);
}

void tCNN::addExample(const tIO& input, const tIO& target)
{
    // Validate the target vector.
    if ((i32)target.size() != m_layers[m_numLayers-1].getOutput().rows())
        throw eInvalidArgument("The target vector must be the same size as the CNN's output.");
    tANN::nLayerType type = m_layers[m_numLayers-1].getLayer().layerType;
    for (size_t i = 0; i < target.size(); i++)
    {
        if (target[i] < s_squash_min(type) || target[i] > s_squash_max(type))
        {
            throw eInvalidArgument("The target vector must be in the range of the "
                                   "top layer's squashing function.");
        }
    }
    if (type == tANN::kLayerTypeSoftmax)
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
    if (input.size() != m_layers[0].getInputSize())
        throw eInvalidArgument("The input vector must be the same size as the CNN's input.");

    // Accumulate this example.
    m_inputAccum.push_back(input);
    m_targetAccum.push_back(target);
}

void tCNN::update()
{
    if (m_inputAccum.size() == 0)
        throw eLogicError("You must give the network some examples before you call update().");

    // Build the input matrix.
    Mat input(m_inputAccum[0].size(), m_inputAccum.size());
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
            m_layers[i].takeInput(m_layers[i-1].getOutput());
    }

    // Backprop.
    {
        assert(m_layers[m_numLayers-1].getOutput().rows() == target.rows());
        assert(m_layers[m_numLayers-1].getOutput().cols() == target.cols());
        m_layers[m_numLayers-1].get_da() = m_layers[m_numLayers-1].getOutput() - target;

        for (u32 i = m_numLayers-1; i > 0; i--)
        {
            m_layers[i].distribute_da(m_layers[i-1].getOutput());
            m_layers[i].accumError();
            m_layers[i].backpropagate(m_layers[i-1].get_da(), m_layers[i-1].getOutput());
            m_layers[i].updateWeights(m_layers[i-1].getOutput());
        }

        m_layers[0].distribute_da(input);
        m_layers[0].accumError();
        m_layers[0].updateWeights(input);
    }
}

void tCNN::evaluate(const tIO& input, tIO& output) const
{
    if (input.size() != m_layers[0].getInputSize())
        throw eInvalidArgument("The input vector must be the same size as the CNN's input.");

    Mat inputMat(input.size(), 1);
    for (size_t i = 0; i < input.size(); i++)
        inputMat(i,0) = input[i];

    m_layers[0].takeInput(inputMat);
    for (u32 i = 1; i < m_numLayers; i++)
        m_layers[i].takeInput(m_layers[i-1].getOutput());

    const Mat& outMat = m_layers[m_numLayers-1].getOutput();
    assert(outMat.rows() > 0);
    assert(outMat.cols() == 1);
    output.resize(outMat.rows());
    for (i32 i = 0; i < outMat.rows(); i++)
        output[i] = outMat(i,0);
}

void tCNN::evaluateBatch(const std::vector<tIO>& inputs,
                               std::vector<tIO>& outputs) const
{
    outputs.resize(inputs.size());
    evaluateBatch(inputs.begin(), inputs.end(), outputs.begin());
}

void tCNN::evaluateBatch(std::vector<tIO>::const_iterator inputStart,
                         std::vector<tIO>::const_iterator inputEnd,
                         std::vector<tIO>::iterator outputStart) const
{
    if (inputStart >= inputEnd)
        throw eInvalidArgument("There must be at least one input example.");

    for (std::vector<tIO>::const_iterator initr = inputStart; initr != inputEnd; initr++)
        if ((initr->size()) != m_layers[0].getInputSize())
            throw eInvalidArgument("All input vectors must be the same size as the CNN's input.");

    // Build the input matrix.
    Mat inputMat(inputStart->size(), inputEnd-inputStart);
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
        m_layers[i].takeInput(m_layers[i-1].getOutput());

    // Capture the output.
    const Mat& outMat = m_layers[m_numLayers-1].getOutput();
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

f64 tCNN::calculateError(const tIO& output, const tIO& target)
{
    if (m_layers[m_numLayers-1].getLayer().layerType == tANN::kLayerTypeSoftmax)
        return crossEntropyCost(output, target);
    else
        return standardSquaredError(output, target);
}

f64 tCNN::calculateError(const std::vector<tIO>& outputs,
                         const std::vector<tIO>& targets)
{
    if (m_layers[m_numLayers-1].getLayer().layerType == tANN::kLayerTypeSoftmax)
        return crossEntropyCost(outputs, targets);
    else
        return standardSquaredError(outputs, targets);
}

void tCNN::reset()
{
    resetWeights();
}

void tCNN::printLearnerInfo(std::ostream& out) const
{
    const int colw = 20;
    std::ostringstream sout;

    sout << "Convolutional Neural Network Info:" << endl;

    // Layer type (and normalizeLayerInput):
    sout << "                 layer type:";
    sout << std::right << std::setw(colw) << "input";
    for (u32 i = 0; i < m_numLayers; i++)
    {
        string print = s_layerTypeToString(m_layers[i].getLayer().layerType);
        if (m_layers[i].getLayer().normalizeLayerInput)
            print += "(norm'd)";
        sout << std::right << std::setw(colw) << print;
    }
    sout << endl;

    // Weight update rule:
    sout << "         weight update rule:";
    sout << std::right << std::setw(colw) << "-";
    for (u32 i = 0; i < m_numLayers; i++)
    {
        std::ostringstream ss;
        ss << s_weightUpRuleToString(m_layers[i].getLayer().weightUpRule);
        switch (m_layers[i].getLayer().weightUpRule)
        {
            case tANN::kWeightUpRuleNone:
            case tANN::kWeightUpRuleRPROP:
                break;
            case tANN::kWeightUpRuleFixedLearningRate:
            case tANN::kWeightUpRuleAdaptiveRates:
            case tANN::kWeightUpRuleRMSPROP:
            case tANN::kWeightUpRuleARMS:
                ss << "(a=" << m_layers[i].getLayer().alpha << ")";
                break;
            case tANN::kWeightUpRuleMomentum:
                ss << "(a=" << m_layers[i].getLayer().alpha
                   << ",v=" << m_layers[i].getLayer().viscosity << ")";
                break;
            default:
                assert(false);
        }
        sout << std::right << std::setw(colw) << ss.str();
    }
    sout << endl;

    // Number of feature maps:
    sout << "        number feature maps:";
    sout << std::right << std::setw(colw) << "-";
    for (u32 i = 0; i < m_numLayers; i++)
        sout << std::right << std::setw(colw) << m_layers[i].getNumFeatureMaps();
    sout << endl;

    // Receptive field sizes:
    sout << "       receptive field size:";
    sout << std::right << std::setw(colw) << "-";
    for (u32 i = 0; i < m_numLayers; i++)
    {
        std::ostringstream ss;
        ss << m_layers[i].getReceptiveFieldWidth() / (i > 0 ? m_layers[i-1].getNumFeatureMaps() : 1)
           << "x"
           << m_layers[i].getReceptiveFieldHeight();
        sout << std::right << std::setw(colw) << ss.str();
    }
    sout << endl;

    // Num free parameters:
    sout << "        num free parameters:";
    sout << std::right << std::setw(colw) << "-";
    for (u32 i = 0; i < m_numLayers; i++)
        sout << std::right << std::setw(colw) << m_layers[i].getNumFreeParameters();
    sout << endl;

    // Num connections:
    sout << "            num connections:";
    sout << std::right << std::setw(colw) << "-";
    for (u32 i = 0; i < m_numLayers; i++)
        sout << std::right << std::setw(colw) << (m_layers[i].getNumFreeParameters() * m_layers[i].getNumReplicas());
    sout << endl;

    // Receptive field step sizes:
    sout << "            field step size:";
    sout << std::right << std::setw(colw) << "-";
    for (u32 i = 0; i < m_numLayers; i++)
    {
        std::ostringstream ss;
        ss << m_layers[i].getStepSizeX() / (i > 0 ? m_layers[i-1].getNumFeatureMaps() : 1)
           << "x"
           << m_layers[i].getStepSizeY();
        sout << std::right << std::setw(colw) << ss.str();
    }
    sout << endl;

    // Network output dimensions:
    sout << "                  out dim's:";
    {
        std::ostringstream ss;
        ss << m_layers[0].getInputWidth() << "x"
           << m_layers[0].getInputHeight();
        sout << std::right << std::setw(colw) << ss.str();
    }
    for (u32 i = 0; i < m_numLayers; i++)
    {
        std::ostringstream ss;
        ss << (m_layers[i].getStepsX()+1);
        ss << "x";
        ss << (m_layers[i].getStepsY()+1);
        sout << std::right << std::setw(colw) << ss.str();
    }
    sout << endl;

    // Pool dimensions:
    sout << "                 pool dim's:";
    sout << std::right << std::setw(colw) << "1x1";
    for (u32 i = 0; i < m_numLayers; i++)
    {
        std::ostringstream ss;
        ss << m_layers[i].getPoolWidth() << "x" << m_layers[i].getPoolHeight();
        sout << std::right << std::setw(colw) << ss.str();
    }
    sout << endl;

    // Network output size:
    sout << "                   out size:";
    sout << std::right << std::setw(colw) << m_layers[0].getInputSize();
    for (u32 i = 0; i < m_numLayers; i++)
        sout << std::right << std::setw(colw) << m_layers[i].getOutput().rows();
    sout << endl;

    sout << endl;

    out << sout.str() << std::flush;
}

string tCNN::learnerInfoString() const
{
    std::ostringstream out;

    // Network output sizes:
    out << "size=";
    out << m_layers[0].getInputSize();
    for (u32 i = 0; i < m_numLayers; i++)
        out << '-' << m_layers[i].getOutput().rows();

    // Layer type (and normalizeLayerInput):
    out << "__type=";
    out << "i";
    for (u32 i = 0; i < m_numLayers; i++)
    {
        out << '-' << s_layerTypeToChar(m_layers[i].getLayer().layerType);
        if (m_layers[i].getLayer().normalizeLayerInput)
            out << 'n';
    }

    // Weight update rule:
    out << "__wrule=";
    out << "i";
    for (u32 i = 0; i < m_numLayers; i++)
    {
        out << '-' << s_weightUpRuleToChar(m_layers[i].getLayer().weightUpRule);
        switch (m_layers[i].getLayer().weightUpRule)
        {
            case tANN::kWeightUpRuleNone:
            case tANN::kWeightUpRuleRPROP:
                break;
            case tANN::kWeightUpRuleFixedLearningRate:
            case tANN::kWeightUpRuleAdaptiveRates:
            case tANN::kWeightUpRuleRMSPROP:
            case tANN::kWeightUpRuleARMS:
                out << m_layers[i].getLayer().alpha;
                break;
            case tANN::kWeightUpRuleMomentum:
                out << m_layers[i].getLayer().alpha << ','
                    << m_layers[i].getLayer().viscosity;
                break;
            default:
                assert(false);
        }
    }

    // Number of feature maps:
    out << "__maps=";
    out << "i";
    for (u32 i = 0; i < m_numLayers; i++)
        out << '-' << m_layers[i].getNumFeatureMaps();

    return out.str();
}

u32 tCNN::getNumLayers() const
{
    return m_numLayers;
}

bool tCNN::isLayerPooled(u32 layerIndex) const
{
    if (layerIndex >= m_numLayers)
        throw eInvalidArgument("No layer with that index.");

    return (m_layers[layerIndex].getPoolWidth() > 1) ||
           (m_layers[layerIndex].getPoolHeight() > 1);
}

bool tCNN::isLayerFullyConnected(u32 layerIndex) const
{
    if (layerIndex >= m_numLayers)
        throw eInvalidArgument("No layer with that index.");

    return (m_layers[layerIndex].getStepsX() == 0) &&
           (m_layers[layerIndex].getStepsY() == 0);
}

u32 tCNN::getNumFeatureMaps(u32 layerIndex) const
{
    if (layerIndex >= m_numLayers)
        throw eInvalidArgument("No layer with that index.");

    return (u32) (m_layers[layerIndex].getNumFeatureMaps());
}

void tCNN::getWeights(u32 layerIndex, u32 mapIndex, tIO& weights) const
{
    if (mapIndex >= getNumFeatureMaps(layerIndex))
        throw eInvalidArgument("No layer/map with that index.");

    const Mat& w = m_layers[layerIndex].getLayer().w;
    assert(w.cols() > 1);
    assert(w.rows() > mapIndex);
    weights.resize(w.cols()-1);
    for (size_t s = 0; s < weights.size(); s++)
        weights[s] = w(mapIndex, s);
}

f64 tCNN::getBias(u32 layerIndex, u32 mapIndex) const
{
    if (mapIndex >= getNumFeatureMaps(layerIndex))
        throw eInvalidArgument("No layer/map with that index.");

    const Mat& w = m_layers[layerIndex].getLayer().w;
    assert(w.cols() > 1);
    assert(w.rows() > mapIndex);
    return w.rightCols(1)(mapIndex,0);
}

u32 tCNN::getNumReplicatedFilters(u32 layerIndex) const
{
    if (layerIndex >= m_numLayers)
        throw eInvalidArgument("No layer with that index.");

    return m_layers[layerIndex].getNumReplicas();
}

f64 tCNN::getOutput(u32 layerIndex, u32 mapIndex, u32 filterIndex,
                    f64* minValue, f64* maxValue) const
{
    if (mapIndex >= getNumFeatureMaps(layerIndex))
        throw eInvalidArgument("No layer/map with that index.");

    if (filterIndex >= getNumReplicatedFilters(layerIndex))
        throw eInvalidArgument("No layer/filter with that index.");

    if (minValue)
        *minValue = s_squash_min(m_layers[layerIndex].getLayer().layerType);
    if (maxValue)
        *maxValue = s_squash_max(m_layers[layerIndex].getLayer().layerType);

    const Mat& a = m_layers[layerIndex].getLayer().a;

    u32 numfilters = m_layers[layerIndex].getNumReplicas();

    if (a.cols() < numfilters)
        throw eInvalidArgument("There is no \"most recent\" output of this filter.");

    assert(a.rows() > mapIndex);
    assert(a.cols()+filterIndex >= numfilters);
    return a(mapIndex,a.cols()+filterIndex-numfilters);
}

void tCNN::getFeatureMapImage(u32 layerIndex, u32 mapIndex,
                              bool color, bool absolute,
                              img::tImage* dest) const
{
    // Get the weights.
    tIO weights;
    getWeights(layerIndex, mapIndex, weights);
    assert(weights.size() > 0);
    u32 width = m_layers[layerIndex].getReceptiveFieldWidth();
    assert(width > 0);
    if (color)
    {
        if ((width % 3) > 0)
            throw eLogicError("Pixels do not align with width of the receptive field.");
        width /= 3;
    }
    assert(width > 0);

    // Use the image creating method in ml::common to do the work.
    un_examplify(weights, color, width, absolute, dest);
}

void tCNN::getOutputImage(u32 layerIndex, u32 mapIndex,
                          bool pooled,
                          img::tImage* dest) const
{
    if (mapIndex >= getNumFeatureMaps(layerIndex))
        throw eInvalidArgument("No layer/map with that index.");

    // Get the weights.
    tIO weights;
    u32 width;
    {
        u32 stride = getNumFeatureMaps(layerIndex);
        const Mat& alloutput = pooled ? m_layers[layerIndex].getOutput()
                                      : m_layers[layerIndex].getRealOutput();

        if (alloutput.cols() == 0)
            throw eInvalidArgument("There is no \"most recent\" output of this filter.");

        for (u32 i = mapIndex; i < (u32)alloutput.rows(); i += stride)
            weights.push_back(alloutput(i,alloutput.cols()-1));
        assert(weights.size() > 0);

        width = pooled ? (m_layers[layerIndex].getStepsX()+1) / m_layers[layerIndex].getPoolWidth()
                       : (m_layers[layerIndex].getStepsX()+1);
        assert(width > 0);
    }

    // Tell un_examplify() about the range of this data.
    // (Note, when creating images from weight values, the range is (-inf, inf), so it
    // is okay to let un_examplify() determine a good range itself, but here we know
    // the range and we want the resulting image to represent the values relative to that
    // range.
    f64 minValue = s_squash_min(m_layers[layerIndex].getLayer().layerType);
    f64 maxValue = s_squash_max(m_layers[layerIndex].getLayer().layerType);

    // Use the image creating method in ml::common to do the work.
    un_examplify(weights, false, width, false, dest, &minValue, &maxValue);
}

void tCNN::pack(iWritable* out) const
{
    rho::pack(out, m_numLayers);
    for (u32 i = 0; i < m_numLayers; i++)
        m_layers[i].pack(out);
    rho::pack(out, m_randWeightMin);
    rho::pack(out, m_randWeightMax);
}

void tCNN::unpack(iReadable* in)
{
    // Try to unpack the network.
    u32 numLayers;
    f64 randWeightMin, randWeightMax;
    rho::unpack(in, numLayers);
    if (numLayers == 0)
        throw eRuntimeError("Invalid CNN stream -- num layers");
    tLayerCNN* layers = new tLayerCNN[numLayers];
    try
    {
        for (u32 i = 0; i < numLayers; i++)
            layers[i].unpack(in);
        rho::unpack(in, randWeightMin);
        rho::unpack(in, randWeightMax);
        if (randWeightMin >= randWeightMax)
            throw eRuntimeError("Invalid CNN stream -- rand range");
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
