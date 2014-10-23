#if __linux__
#pragma GCC optimize 3
#endif

#include <ml/common.h>
#include <ml/tANN.h>
#include <ml/tCNN.h>

#include <rho/img/tCanvas.h>
#include <rho/algo/vector_util.h>
#include <rho/sync/tTimer.h>

#include <cassert>
#include <iomanip>
#include <sstream>


namespace ml
{


f64 logistic_function(f64 z)
{
    return (1.0 / (1.0 + std::exp(-z)));
}

f64 derivative_of_logistic_function(f64 z)
{
    f64 y = logistic_function(z);
    f64 slope = (y * (1.0 - y));
    slope = std::max(slope, 1e-5);    // <-- Experimental
    return slope;
}

f64 inverse_of_logistic_function(f64 y)
{
    if (y < 0.0001) y = 0.0001;
    if (y > 0.9999) y = 0.9999;
    return -std::log((1.0 / y) - 1.0);
}

f64 logistic_function_min()
{
    return 0.0;
}

f64 logistic_function_max()
{
    return 1.0;
}


f64 hyperbolic_function(f64 z)
{
    // Recommended by: "Efficient BackProp" (LeCun et al.)
    return 1.7159 * std::tanh(2.0/3.0 * z);
}

f64 derivative_of_hyperbolic_function(f64 z)
{
    f64 s = 1.0 / std::cosh(2.0/3.0 * z);
    return 1.14393 * s * s;
}

f64 inverse_of_hyperbolic_function(f64 y)
{
    if (y < -1.71589) y = -1.71589;
    if (y > 1.71589) y = 1.71589;
    return 1.5 * atanh(0.582785 * y);
}

f64 hyperbolic_function_min()
{
    return -1.7159;
}

f64 hyperbolic_function_max()
{
    return 1.7159;
}


tIO examplify(u32 highDimension, u32 numDimensions)
{
    if (highDimension >= numDimensions)
        throw eInvalidArgument("highDimension must be < numDimensions");
    tIO target(numDimensions, 0.0);
    target[highDimension] = 1.0;
    return target;
}

u32 un_examplify(const tIO& output, f64* error)
{
    if (output.size() == 0)
        throw eInvalidArgument("The output vector must have at least one dimension!");
    u32 maxindex = 0;
    for (size_t i = 1; i < output.size(); i++)
        if (output[i] > output[maxindex])
            maxindex = (u32)i;
    if (error)
        *error = standardSquaredError(output, examplify(maxindex, (u32)output.size()));
    return maxindex;
}

tIO examplify(const img::tImage* image)
{
    if (image->bufUsed() == 0)
        throw eInvalidArgument("The example image must have at least one pixel in it!");
    const u8* buf = image->buf();
    tIO input(image->bufUsed(), 0.0);
    for (u32 i = 0; i < image->bufUsed(); i++)
        input[i] = buf[i] / 255.0;
    return input;
}

void un_examplify(const tIO& io, bool color, u32 width,
                  bool absolute, img::tImage* dest,
                  const f64* minValue, const f64* maxValue)
{
    if (io.size() == 0)
        throw eInvalidArgument("The example io must have at least one dimension!");
    if (width == 0)
        throw eInvalidArgument("Width may not be zero.");

    // Create a copy of io that can be modified.
    std::vector<f64> weights = io;

    // Normalize the weights to [0.0, 255.0].
    f64 maxval;
    f64 minval;
    if (minValue && maxValue)
    {
        maxval = *maxValue;
        minval = *minValue;
        if (minval > maxval)
            throw eInvalidArgument("The minValue must be less than or equal to the maxValue.");
        for (u32 i = 0; i < weights.size(); i++)
            if (weights[i] < minval || weights[i] > maxval)
                throw eInvalidArgument("The minValue and maxValue cannot be true given this input vector.");
    }
    else
    {
        maxval = weights[0];
        minval = weights[0];
        for (u32 i = 1; i < weights.size(); i++)
        {
            maxval = std::max(maxval, weights[i]);
            minval = std::min(minval, weights[i]);
        }
        if (maxval == minval) maxval += 0.000001;
    }
    f64 absmax = std::max(std::fabs(maxval), std::fabs(minval));
    if (color)
    {
        if (absolute)
        {
            for (u32 i = 0; i < weights.size(); i++)
                weights[i] = (std::fabs(weights[i]) / absmax) * 255.0;
        }
        else
        {
            for (u32 i = 0; i < weights.size(); i++)
            {
                f64 val = ((weights[i] - minval) / (maxval - minval)) * 255.0;
                weights[i] = val;
            }
        }
    }

    // Calculate some stuff.
    u32 pixWidth = color ? 3 : 1;
    if ((weights.size() % pixWidth) > 0)
        throw eLogicError("Pixels do not align with the number of weights.");
    u32 numPix = (u32) weights.size() / pixWidth;
    if ((numPix % width) > 0)
        throw eLogicError("Cannot build image of that width. Last row not filled.");
    u32 height = numPix / width;

    // Create the image.
    dest->setFormat(img::kRGB24);
    dest->setBufSize(width*height*3);
    dest->setBufUsed(width*height*3);
    dest->setWidth(width);
    dest->setHeight(height);
    u8* buf = dest->buf();
    u32 bufIndex = 0;
    u32 wIndex = 0;
    for (u32 i = 0; i < height; i++)
    {
        for (u32 j = 0; j < width; j++)
        {
            if (color)
            {
                buf[bufIndex++] = (u8) weights[wIndex++];
                buf[bufIndex++] = (u8) weights[wIndex++];
                buf[bufIndex++] = (u8) weights[wIndex++];
            }
            else
            {
                u8 r = 0;     // <-- used if the weight is negative
                u8 g = 0;     // <-- used if the weight is positive
                u8 b = 0;     // <-- not used

                f64 w = weights[wIndex++];

                if (w > 0.0)
                    g = (u8)(w / absmax * 255.0);

                if (w < 0.0)
                    r = (u8)(-w / absmax * 255.0);

                buf[bufIndex++] = r;
                buf[bufIndex++] = g;
                buf[bufIndex++] = b;
            }
        }
    }
}

void zscore(std::vector<tIO>& inputs, u32 dStart, u32 dEnd)
{
    // Make sure all the input looks okay.
    if (inputs.size() == 0)
        throw eInvalidArgument("There must be at least one training input!");
    for (size_t i = 1; i < inputs.size(); i++)
    {
        if (inputs[i].size() != inputs[0].size())
        {
            throw eInvalidArgument("Every training input must have the same dimensionality!");
        }
    }

    // For every dimension, we'll need to create a vector of all that dimensions examples.
    tIO dim(inputs.size(), 0.0);

    // For every dimension...
    for (size_t d = dStart; d < inputs[0].size() && d < dEnd; d++)
    {
        // ... Fill 'dim' with that dimension
        for (size_t i = 0; i < inputs.size(); i++)
            dim[i] = inputs[i][d];

        // ... Calculate the mean and stddev
        f64 mean = algo::mean(dim);
        f64 stddev = algo::stddev(dim);

        // ... Normalize that dimension
        if (stddev != 0.0)
        {
            for (size_t i = 0; i < inputs.size(); i++)
                inputs[i][d] = (inputs[i][d] - mean) / stddev;
        }
        else
        {
            for (size_t i = 0; i < inputs.size(); i++)
                inputs[i][d] = 0.0;
        }
    }
}

void zscore(std::vector<tIO>& trainingInputs, std::vector<tIO>& testInputs)
{
    // Make sure all the input looks okay.
    if (trainingInputs.size() == 0)
        throw eInvalidArgument("There must be at least one training input!");
    if (testInputs.size() == 0)
        throw eInvalidArgument("There must be at least one test input!");
    for (size_t i = 1; i < trainingInputs.size(); i++)
    {
        if (trainingInputs[i].size() != trainingInputs[0].size())
        {
            throw eInvalidArgument("Every training input must have the same dimensionality!");
        }
    }
    for (size_t i = 1; i < testInputs.size(); i++)
    {
        if (testInputs[i].size() != testInputs[0].size())
        {
            throw eInvalidArgument("Every test input must have the same dimensionality!");
        }
    }
    if (trainingInputs[0].size() != testInputs[0].size())
        throw eInvalidArgument("The training and test examples must all have the same dimensionality!");

    // For every dimension, we'll need to create a vector of all that dimensions examples.
    tIO dim(trainingInputs.size(), 0.0);

    // For every dimension...
    for (size_t d = 0; d < trainingInputs[0].size(); d++)
    {
        // ... Fill 'dim' with that dimension
        for (size_t i = 0; i < trainingInputs.size(); i++)
            dim[i] = trainingInputs[i][d];

        // ... Calculate the mean and stddev
        f64 mean = algo::mean(dim);
        f64 stddev = algo::stddev(dim);

        // ... Normalize that dimension
        if (stddev != 0.0)
        {
            for (size_t i = 0; i < trainingInputs.size(); i++)
                trainingInputs[i][d] = (trainingInputs[i][d] - mean) / stddev;
            for (size_t i = 0; i < testInputs.size(); i++)
                testInputs[i][d] = (testInputs[i][d] - mean) / stddev;
        }
        else
        {
            for (size_t i = 0; i < trainingInputs.size(); i++)
                trainingInputs[i][d] = 0.0;
            for (size_t i = 0; i < testInputs.size(); i++)
                testInputs[i][d] = 0.0;
        }
    }
}


f64 standardSquaredError(const tIO& output, const tIO& target)
{
    if (output.size() != target.size())
        throw eInvalidArgument(
                "The output vector must have the same dimensionality as the target vector!");
    if (output.size() == 0)
        throw eInvalidArgument("The output and target vectors must have at least one dimension!");
    f64 error = 0.0;
    for (size_t i = 0; i < output.size(); i++)
        error += (output[i]-target[i]) * (output[i]-target[i]);
    return 0.5*error;
}

f64 standardSquaredError(const std::vector<tIO>& outputs,
                         const std::vector<tIO>& targets)
{
    if (outputs.size() != targets.size())
    {
        throw eInvalidArgument("The number of examples in outputs and targets must "
                "be the same!");
    }

    if (outputs.size() == 0)
    {
        throw eInvalidArgument("There must be at least one output/target pair!");
    }

    for (size_t i = 0; i < outputs.size(); i++)
    {
        if (outputs[i].size() != targets[i].size() ||
            outputs[i].size() != outputs[0].size())
        {
            throw eInvalidArgument("Every output/target pair must have the same dimensionality!");
        }
    }

    if (outputs[0].size() == 0)
    {
        throw eInvalidArgument("The output/target pairs must have at least one dimension!");
    }

    f64 error = 0.0;
    for (size_t i = 0; i < outputs.size(); i++)
        error += standardSquaredError(outputs[i], targets[i]);
    return error / ((f64)outputs.size());
}

f64 crossEntropyCost(const tIO& output, const tIO& target)
{
    if (output.size() != target.size())
        throw eInvalidArgument(
                "The output vector must have the same dimensionality as the target vector!");
    if (output.size() == 0)
        throw eInvalidArgument("The output and target vectors must have at least one dimension!");
    f64 osum = 0.0;
    f64 tsum = 0.0;
    for (size_t i = 0; i < output.size(); i++)
    {
        if (output[i] > 1.0)
            throw eInvalidArgument("The output value cannot be >1.0 when it represents a probability.");
        if (output[i] < 0.0)
            throw eInvalidArgument("The output value cannot be <0.0 when it represents a probability.");
        if (target[i] > 1.0)
            throw eInvalidArgument("The target value cannot be >1.0 when it represents a probability.");
        if (target[i] < 0.0)
            throw eInvalidArgument("The target value cannot be <0.0 when it represents a probability.");
        osum += output[i];
        tsum += target[i];
    }
    if (osum > 1.0001 || osum < 0.9999)
        throw eInvalidArgument("The sum of the outputs must be 1.0.");
    if (tsum > 1.0001 || tsum < 0.9999)
        throw eInvalidArgument("The sum of the targets must be 1.0.");
    f64 error = 0.0;
    for (size_t i = 0; i < output.size(); i++)
        if (target[i] > 0.0)
            error += target[i] * std::log(output[i]);
    return -error;
}

f64 crossEntropyCost(const std::vector<tIO>& outputs,
                     const std::vector<tIO>& targets)
{
    if (outputs.size() != targets.size())
    {
        throw eInvalidArgument("The number of examples in outputs and targets must "
                "be the same!");
    }

    if (outputs.size() == 0)
    {
        throw eInvalidArgument("There must be at least one output/target pair!");
    }

    for (size_t i = 0; i < outputs.size(); i++)
    {
        if (outputs[i].size() != targets[i].size() ||
            outputs[i].size() != outputs[0].size())
        {
            throw eInvalidArgument("Every output/target pair must have the same dimensionality!");
        }
    }

    if (outputs[0].size() == 0)
    {
        throw eInvalidArgument("The output/target pairs must have at least one dimension!");
    }

    f64 error = 0.0;
    for (size_t i = 0; i < outputs.size(); i++)
        error += crossEntropyCost(outputs[i], targets[i]);
    return error / ((f64)outputs.size());
}

f64 rmsError(const std::vector<tIO>& outputs,
             const std::vector<tIO>& targets)
{
    f64 sqrdError = standardSquaredError(outputs, targets);
    return std::sqrt(sqrdError * 2.0 / ((f64)outputs[0].size()));
}


void buildConfusionMatrix(const std::vector<tIO>& outputs,
                          const std::vector<tIO>& targets,
                                tConfusionMatrix& confusionMatrix)
{
    if (outputs.size() != targets.size())
    {
        throw eInvalidArgument("The number of examples in outputs and targets must "
                "be the same!");
    }

    if (outputs.size() == 0)
    {
        throw eInvalidArgument("There must be at least one output/target pair!");
    }

    for (size_t i = 0; i < outputs.size(); i++)
    {
        if (outputs[i].size() != targets[i].size() ||
            outputs[i].size() != outputs[0].size())
        {
            throw eInvalidArgument("Every output/target pair must have the same dimensionality!");
        }
    }

    if (outputs[0].size() == 0)
    {
        throw eInvalidArgument("The output/target pairs must have at least one dimension!");
    }

    confusionMatrix.resize(targets[0].size());
    for (size_t i = 0; i < confusionMatrix.size(); i++)
        confusionMatrix[i] = std::vector<u32>(outputs[0].size(), 0);

    for (size_t i = 0; i < outputs.size(); i++)
    {
        u32 target = un_examplify(targets[i]);
        u32 output = un_examplify(outputs[i]);
        confusionMatrix[target][output]++;
    }
}

static
void s_fill_cell(const std::vector<u32>& indices, const std::vector<tIO>& inputs,
                 bool color, u32 width, bool absolute,
                 u32 boxWidth, double ox, double oy,
                 algo::iLCG& lcg, img::tCanvas& canvas)
{
    const u32 kPadding = 5;

    img::tImage image;

    u8 bgColor[3] = { 255, 255, 255 };    // white
    img::tCanvas subcanvas(img::kRGB24, bgColor, 3);
    u32 subcanvX = 0;
    u32 subcanvY = 0;
    bool subcanvFull = false;

    for (size_t i = 0; i < indices.size(); i++)
    {
        un_examplify(inputs[indices[i]], color, width, absolute, &image);

        if (!subcanvFull)
        {
            subcanvas.drawImage(&image, subcanvX, subcanvY);
            subcanvX += image.width() + kPadding;
            if (subcanvX + image.width() + kPadding > boxWidth)
            {
                subcanvX = 0;
                subcanvY += image.height() + kPadding;
                if (subcanvY + image.height() + kPadding > boxWidth)
                {
                    subcanvFull = true;
                    subcanvas.genImage(&image);
                    f64 rx = ox + boxWidth / 2.0 - image.width() / 2.0;
                    f64 ry = oy + boxWidth / 2.0 - image.height() / 2.0;
                    canvas.drawImage(&image, (i32) round(rx), (i32) round(ry));
                }
            }
        }

        else
        {
            f64 rx = ((f64)lcg.next()) / ((f64)lcg.randMax()) * (boxWidth-width) + ox;
            f64 ry = ((f64)lcg.next()) / ((f64)lcg.randMax()) * (boxWidth-width) + oy;
            canvas.drawImage(&image, (i32) round(rx), (i32) round(ry));
        }
    }

    if (!subcanvFull)
    {
        subcanvas.genImage(&image);
        f64 rx = ox + boxWidth / 2.0 - image.width() / 2.0;
        f64 ry = oy + boxWidth / 2.0 - image.height() / 2.0;
        canvas.drawImage(&image, (i32) round(rx), (i32) round(ry));
    }
}

static
void s_drawGrid(img::tCanvas& canvas, u32 gridSize, u32 distBetweenLines)
{
    {
        img::tImage horiz;
        horiz.setFormat(img::kRGB24);
        horiz.setWidth(gridSize*distBetweenLines);
        horiz.setHeight(1);
        horiz.setBufSize(horiz.width() * horiz.height() * 3);
        horiz.setBufUsed(horiz.bufSize());
        for (u32 i = 0; i < horiz.bufUsed(); i++) horiz.buf()[i] = 0;  // <-- makes the lines black
        for (u32 i = 0; i <= gridSize; i++)
            canvas.drawImage(&horiz, 0, i*distBetweenLines);
    }

    {
        img::tImage vert;
        vert.setFormat(img::kRGB24);
        vert.setWidth(1);
        vert.setHeight(gridSize*distBetweenLines);
        vert.setBufSize(vert.width() * vert.height() * 3);
        vert.setBufUsed(vert.bufSize());
        for (u32 i = 0; i < vert.bufUsed(); i++) vert.buf()[i] = 0;  // <-- makes the lines black
        for (u32 i = 0; i <= gridSize; i++)
            canvas.drawImage(&vert, i*distBetweenLines, 0);
    }
}

void buildVisualConfusionMatrix(const std::vector<tIO>& inputs,
                                bool color, u32 width, bool absolute,
                                const std::vector<tIO>& outputs,
                                const std::vector<tIO>& targets,
                                      img::tImage* dest,
                                u32 cellWidthMultiplier)
{
    if (outputs.size() != targets.size())
    {
        throw eInvalidArgument("The number of examples in outputs and targets must "
                "be the same!");
    }

    if (outputs.size() != inputs.size())
    {
        throw eInvalidArgument("The number of examples in outputs and inputs must "
                "be the same!");
    }

    if (outputs.size() == 0)
    {
        throw eInvalidArgument("There must be at least one output/target pair!");
    }

    for (size_t i = 0; i < outputs.size(); i++)
    {
        if (outputs[i].size() != targets[i].size() ||
            outputs[i].size() != outputs[0].size())
        {
            throw eInvalidArgument("Every output/target pair must have the same dimensionality!");
        }
    }

    if (outputs[0].size() == 0)
    {
        throw eInvalidArgument("The output/target pairs must have at least one dimension!");
    }

    u32 numClasses = (u32) targets[0].size();      // same as outputs[0].size()

    std::vector< std::vector< std::vector<u32> > > holding(numClasses,
            std::vector< std::vector<u32> >(numClasses, std::vector<u32>()));

    for (size_t i = 0; i < outputs.size(); i++)    // same as targets.size()
    {
        u32 target = un_examplify(targets[i]);
        u32 output = un_examplify(outputs[i]);
        holding[target][output].push_back((u32)i);
    }

    algo::tKnuthLCG lcg;

    u32 boxWidth = cellWidthMultiplier * width;
    u8 bgColor[3] = { 255, 255, 255 };    // white
    img::tCanvas canvas(img::kRGB24, bgColor, 3);

    for (size_t i = 0; i < holding.size(); i++)
    {
        for (size_t j = 0; j < holding[i].size(); j++)
        {
            s_fill_cell(holding[i][j], inputs,
                        color, width, absolute,
                        boxWidth, (f64)(j*boxWidth), (f64)(i*boxWidth),
                        lcg, canvas);
        }
    }

    canvas.expandToIncludePoint(0, 0);
    canvas.expandToIncludePoint(numClasses*boxWidth, numClasses*boxWidth);
    s_drawGrid(canvas, numClasses, boxWidth);
    canvas.genImage(dest);
}

static
void checkConfusionMatrix(const tConfusionMatrix& confusionMatrix)
{
    if (confusionMatrix.size() == 0)
        throw eInvalidArgument("Invalid confusion matrix");

    for (size_t i = 0; i < confusionMatrix.size(); i++)
    {
        if (confusionMatrix[i].size() != confusionMatrix.size())
            throw eInvalidArgument("Invalid confusion matrix");
    }
}

static
void printDashes(const tConfusionMatrix& confusionMatrix, std::ostream& out, u32 s, u32 w)
{
    for (u32 i = 1; i < s; i++)
        out << " ";
    out << "+";
    for (size_t j = 0; j < confusionMatrix[0].size(); j++)
    {
        for (u32 i = 1; i < w; i++)
            out << "-";
        out << "+";
    }
    out << std::endl;
}

void print(const tConfusionMatrix& confusionMatrix, std::ostream& out)
{
    checkConfusionMatrix(confusionMatrix);

    u32 s = 14;
    u32 w = 10;

    out << "                   predicted" << std::endl;

    printDashes(confusionMatrix, out, s, w);

    for (size_t i = 0; i < confusionMatrix.size(); i++)
    {
        if (i == confusionMatrix.size()/2)
            out << "  correct    |";
        else
            out << "             |";
        for (size_t j = 0; j < confusionMatrix[i].size(); j++)
        {
            out << " " << std::right << std::setw(w-3) << confusionMatrix[i][j] << " |";
        }
        out << std::endl;
    }

    printDashes(confusionMatrix, out, s, w);

    out << std::endl;
}

f64  errorRate(const tConfusionMatrix& confusionMatrix)
{
    checkConfusionMatrix(confusionMatrix);

    u32 total = 0;
    for (size_t i = 0; i < confusionMatrix.size(); i++)
        for (size_t j = 0; j < confusionMatrix[i].size(); j++)
            total += confusionMatrix[i][j];
    u32 correct = 0;
    for (size_t i = 0; i < confusionMatrix.size(); i++)
        correct += confusionMatrix[i][i];
    return ((f64)(total - correct)) / total;
}

f64  accuracy(const tConfusionMatrix& confusionMatrix)
{
    checkConfusionMatrix(confusionMatrix);

    u32 total = 0;
    for (size_t i = 0; i < confusionMatrix.size(); i++)
        for (size_t j = 0; j < confusionMatrix[i].size(); j++)
            total += confusionMatrix[i][j];
    u32 correct = 0;
    for (size_t i = 0; i < confusionMatrix.size(); i++)
        correct += confusionMatrix[i][i];
    return ((f64)correct) / total;
}

f64  precision(const tConfusionMatrix& confusionMatrix)
{
    checkConfusionMatrix(confusionMatrix);

    if (confusionMatrix.size() != 2)
        throw eInvalidArgument("Precision is only defined for boolean classification.");

    f64 tp = (f64) confusionMatrix[1][1];
    f64 fp = (f64) confusionMatrix[0][1];
    //f64 tn = (f64) confusionMatrix[0][0];
    //f64 fn = (f64) confusionMatrix[1][0];

    return tp / (tp + fp);
}

f64  recall(const tConfusionMatrix& confusionMatrix)
{
    checkConfusionMatrix(confusionMatrix);

    if (confusionMatrix.size() != 2)
        throw eInvalidArgument("Recall is only defined for boolean classification.");

    f64 tp = (f64) confusionMatrix[1][1];
    //f64 fp = (f64) confusionMatrix[0][1];
    //f64 tn = (f64) confusionMatrix[0][0];
    f64 fn = (f64) confusionMatrix[1][0];

    return tp / (tp + fn);
}


bool train(iLearner* learner, const std::vector<tIO>& inputs,
                              const std::vector<tIO>& targets,
                              u32 batchSize,
                              iTrainObserver* trainObserver)
{
    if (inputs.size() != targets.size())
    {
        throw eInvalidArgument("The number of examples in inputs and targets must "
                "be the same!");
    }

    if (inputs.size() == 0)
    {
        throw eInvalidArgument("There must be at least one input/target pair!");
    }

    for (size_t i = 1; i < inputs.size(); i++)
    {
        if (inputs[i].size() != inputs[0].size())
        {
            throw eInvalidArgument("Every input must have the same dimensionality!");
        }
    }

    for (size_t i = 1; i < targets.size(); i++)
    {
        if (targets[i].size() != targets[0].size())
        {
            throw eInvalidArgument("Every target must have the same dimensionality!");
        }
    }

    if (batchSize == 0)
    {
        throw eInvalidArgument("batchSize must be positive!");
    }

    std::vector<tIO> mostRecentBatch(batchSize);
    u32 batchCounter = 0;

    for (size_t i = 0; i < inputs.size(); i++)
    {
        learner->addExample(inputs[i], targets[i]);
        mostRecentBatch[batchCounter] = inputs[i];
        batchCounter++;
        if (batchCounter == batchSize)
        {
            learner->update();
            if (trainObserver && !trainObserver->didUpdate(learner, mostRecentBatch))
                return false;
            batchCounter = 0;
        }
    }
    if (batchCounter > 0)
    {
        learner->update();
        mostRecentBatch.resize(batchCounter);
        if (trainObserver && !trainObserver->didUpdate(learner, mostRecentBatch))
            return false;
    }

    return true;
}

void evaluate(iLearner* learner, const std::vector<tIO>& inputs,
                                       std::vector<tIO>& outputs,
                                 u32 batchSize)
{
    if (inputs.size() == 0)
        throw eInvalidArgument("There must be at least one input vector!");
    if (batchSize == 0)
        throw eInvalidArgument("The batch size must be positive.");
    outputs.resize(inputs.size());
    for (size_t i = 0; i < inputs.size(); i += batchSize)
    {
        size_t sizeHere = std::min((size_t)batchSize, inputs.size()-i);
        learner->evaluateBatch(inputs.begin()+i,
                               inputs.begin()+i+sizeHere,
                               outputs.begin()+i);
    }
}

/*
 * This function is used "deinterlace" ("deinterleave" is the correct
 * term, actually) a vector of repeating component.
 *
 * For example, say you have a vector with the contents: a1b2c3d4
 * And you want to convert that to a vector: abcd1234
 * To do that call this function with numComponents=2 and unitLength=1.
 *
 * Or, say you have a vector with the contents: ab12cd34ef56
 * And you want to convert that to a vector: abcdef123456
 * To do that call this function with numComponents=2 and unitLength=2.
 *
 * Or, say you have a vector with the contents: RGBRGBRGBRGB
 * And you want to convert that to a vector: RRRRGGGGBBBB
 * To do that call this function with numComponents=3 and unitLength=1.
 *
 * 'output' must be allocated by the caller, and of course delete by
 * the caller as well.
 */
template <class T>
void deinterlace(const T* input, T* output, u32 arrayLen, u32 numComponents, u32 unitLength=1)
{
    assert((arrayLen % unitLength) == 0);
    u32 numUnits = arrayLen / unitLength;

    assert((numUnits % numComponents) == 0);
    u32 groupSize = numUnits / numComponents;

    u32 stride = groupSize * unitLength;

    u32 s = 0;

    for (u32 g = 0; g < groupSize; g++)
    {
        u32 d = g * unitLength;

        for (u32 c = 0; c < numComponents; c++)
        {
            for (u32 u = 0; u < unitLength; u++)
                output[d+u] = input[s+u];

            s += unitLength;
            d += stride;
        }
    }
}

/*
 * This function is used to re-order the components of a CNN hidden layer weight
 * image. The issue is that in the image, weights to each of the previous receptive
 * fields are interlaced together, but when we view that image we'd rather see
 * the weights to the same previous receptive field together. So this function
 * fixes that.
 */
static
void s_fixHiddenLayerImage(img::tImage& image, u32 numComponents)
{
    u32 width = 0;
    u32 unitLength = 0;

    switch (image.format())
    {
        case img::kRGB24:
            width = image.width() * 3;
            unitLength = 3;
            break;

        case img::kGrey:
            width = image.width();
            unitLength = 1;
            break;

        default:
            assert(false);
            break;
    }

    u8* temp = new u8[width];

    for (u32 h = 0; h < image.height(); h++)
    {
        u8* row = image[h][0];
        deinterlace(row, temp, width, numComponents, unitLength);
        for (u32 i = 0; i < width; i++)
            row[i] = temp[i];
    }

    delete [] temp;
}

static
void s_drawLayer(const tCNN* cnn, u32 layerIndex,
                 bool color, bool absolute,
                 u32& currX, img::tCanvas& canvas)
{
    const u32 padding = 10;
    const u32 weigthImageScale = 5;

    u32 currY = padding;

    for (u32 n = 0; n < cnn->getNumFeatureMaps(layerIndex); n++)
    {
        u32 heightHere = 0;
        u32 xHere = currX;

        if (cnn->isLayerFullyConnected(layerIndex))
        {
            {
                img::tImage wImage;
                cnn->getFeatureMapImage(layerIndex, n, layerIndex == 0 ? color : false, absolute, &wImage);
                if (layerIndex > 0)
                    s_fixHiddenLayerImage(wImage, cnn->getNumFeatureMaps(layerIndex-1));
                img::tImage wImageScaled;
                wImage.scale(wImage.width()*weigthImageScale, wImage.height()*weigthImageScale, &wImageScaled);
                canvas.drawImage(&wImageScaled, xHere, currY);
                heightHere = wImageScaled.height();
                xHere += wImageScaled.width();
            }
            {
                xHere += padding;
                img::tImage image;
                image.setFormat(img::kRGB24); image.setWidth(padding*2); image.setHeight(heightHere);
                image.setBufSize(image.width() * image.height() * 3);
                image.setBufUsed(image.bufSize());
                f64 minpossible, maxpossible;
                f64 val = cnn->getOutput(layerIndex, n, 0, &minpossible, &maxpossible);
                for (u32 i = 0; i < image.bufUsed(); i += 3)
                {
                    u8 r = 0;   // <-- used if val is negative
                    u8 g = 0;   // <-- used if val is positive
                    u8 b = 0;   // <-- not used
                    if (val < 0.0)
                        r = (u8) (val / minpossible * 255.0);
                    else if (val > 0.0)
                        g = (u8) (val / maxpossible * 255.0);
                    image.buf()[i+0] = r;
                    image.buf()[i+1] = g;
                    image.buf()[i+2] = b;
                }
                canvas.drawImage(&image, xHere, currY);
                xHere += image.width();
            }
        }

        else
        {
            {
                img::tImage wImage;
                cnn->getFeatureMapImage(layerIndex, n, layerIndex == 0 ? color : false, absolute, &wImage);
                if (layerIndex > 0)
                    s_fixHiddenLayerImage(wImage, cnn->getNumFeatureMaps(layerIndex-1));
                img::tImage wImageScaled;
                wImage.scale(wImage.width()*weigthImageScale, wImage.height()*weigthImageScale, &wImageScaled);
                canvas.drawImage(&wImageScaled, xHere, currY);
                heightHere = std::max(heightHere, wImageScaled.height());
                xHere += wImageScaled.width();
            }
            {
                xHere += padding;
                img::tImage layerImage;
                cnn->getOutputImage(layerIndex, n, false, &layerImage);   // <-- false for "not pooled"
                canvas.drawImage(&layerImage, xHere, currY);
                heightHere = std::max(heightHere, layerImage.height());
                xHere += layerImage.width();
            }
            if (cnn->isLayerPooled(layerIndex))
            {
                xHere += padding;
                img::tImage layerImage;
                cnn->getOutputImage(layerIndex, n, true, &layerImage);   // <-- true for "pooled"
                canvas.drawImage(&layerImage, xHere, currY);
                heightHere = std::max(heightHere, layerImage.height());
                xHere += layerImage.width();
            }
        }

        currY += heightHere + padding;
        if (n+1 == cnn->getNumFeatureMaps(layerIndex))
            currX = xHere;

        canvas.expandToIncludePoint(currX, currY);
    }
}

void visualize(iLearner* learner, const tIO& example,
               bool color, u32 width, bool absolute,
               img::tImage* dest)
{
    tANN* ann = dynamic_cast<tANN*>(learner);
    tCNN* cnn = dynamic_cast<tCNN*>(learner);

    if (ann)
    {
        tIO output;
        ann->evaluate(example, output);

        u8 bgColor[3] = { 0, 0, 205 };    // "Medium Blue" from http://www.tayloredmktg.com/rgb/
        img::tCanvas canvas(img::kRGB24, bgColor, 3);

        u32 numHoriz = (u32) std::ceil(std::sqrt(3.0 * ann->getNumNeuronsInLayer(0)));
        if (numHoriz < 2) numHoriz = 2;
        u32 horizCount = 0;
        u32 currWidth = 0;
        u32 currHeight = 0;

        img::tImage image;
        un_examplify(example, color, width, absolute, &image);
        canvas.drawImage(&image, currWidth, currHeight);
        horizCount++;
        currWidth += image.width();

        for (u32 n = 0; n < ann->getNumNeuronsInLayer(0); n++)
        {
            ann->getImage(0, n, color, width, absolute, &image);
            canvas.drawImage(&image, currWidth, currHeight);
            currWidth += image.width();
            if (++horizCount == numHoriz)
            {
                horizCount = 0;
                currWidth = 0;
                currHeight += image.height();
            }
        }

        canvas.genImage(dest);
    }

    else if (cnn)
    {
        tIO output;
        cnn->evaluate(example, output);

        u8 bgColor[3] = { 0, 0, 205 };    // "Medium Blue" from http://www.tayloredmktg.com/rgb/
        img::tCanvas canvas(img::kRGB24, bgColor, 3);

        const u32 padding = 50;
        u32 currX = 10;

        img::tImage exampleImage;
        un_examplify(example, color, width, absolute, &exampleImage);
        canvas.drawImage(&exampleImage, currX, currX);
        currX += exampleImage.width() + padding;

        for (u32 i = 0; i < cnn->getNumLayers(); i++)
        {
            s_drawLayer(cnn, i, color, absolute, currX, canvas);
            currX += padding;
        }

        canvas.expandToIncludePoint(0, 0);
        canvas.expandToIncludePoint(currX, 0);
        canvas.genImage(dest);
    }

    else
    {
        throw eNotImplemented("Is there some type of learner that I don't know how to draw?");
    }
}


static
u32  s_ezTrain(iLearner* learner,       std::vector< tIO >& trainInputs,
                                        std::vector< tIO >& trainTargets,
                                  const std::vector< tIO >& testInputs,
                                  const std::vector< tIO >& testTargets,
                                  u32 batchSize,
                                  iEZTrainObserver* trainObserver,
                                  u32 foldIndex, u32 numFolds)
{
    if (trainInputs.size() != trainTargets.size())
        throw eInvalidArgument("The number of training inputs does not match the number of training targets.");
    if (testInputs.size() != testTargets.size())
        throw eInvalidArgument("The number of testing inputs does not match the number of testing targets.");
    if (trainInputs.size() == 0 || testInputs.size() == 0)
        throw eInvalidArgument("The training and test sets must each be non-empty.");
    if (batchSize == 0)
        throw eInvalidArgument("The batch size must be non-zero.");

    u64 trainStartTime = sync::tTimer::usecTime();

    std::vector<tIO> trainOutputs;
    tConfusionMatrix trainCM;

    std::vector<tIO> testOutputs;
    tConfusionMatrix testCM;

    algo::tKnuthLCG lcg;

    for (u32 epochs = 0; true; epochs++)
    {
        // Note the start time of this epoch.
        u64 startTime = sync::tTimer::usecTime();

        // Train if this is not the zero'th epoch. This is so that the user will get a
        // callback before any training has happened, so that the user knows what the
        // initial state of the learner looks like.
        if (epochs > 0)
        {
            if (! train(learner, trainInputs, trainTargets,
                        batchSize, trainObserver))
            {
                if (trainObserver)
                {
                    f64 trainElapsedTime = (f64)(sync::tTimer::usecTime() - trainStartTime);
                    trainElapsedTime /= 1000000;  // usecs to secs
                    trainObserver->didFinishTraining(learner, epochs-1, foldIndex, numFolds,
                                                     trainInputs, trainTargets, testInputs, testTargets,
                                                     trainElapsedTime);
                }
                return epochs-1;
            }

            // Shuffle the training data for the next iteration.
            algo::shuffle(trainInputs, trainTargets, lcg);
        }

        // Call the epoch observer.
        if (trainObserver)
        {
            // Evaluate the learner using the training set.
            evaluate(learner, trainInputs, trainOutputs, batchSize);
            buildConfusionMatrix(trainOutputs, trainTargets, trainCM);

            // Evaluate the learner using the test set.
            evaluate(learner, testInputs, testOutputs, batchSize);
            buildConfusionMatrix(testOutputs, testTargets, testCM);

            // Calculate the elapsed time.
            f64 elapsedTime = (f64)(sync::tTimer::usecTime() - startTime);
            elapsedTime /= 1000000;  // usecs to secs

            if (! trainObserver->didFinishEpoch(learner,
                                                epochs,
                                                foldIndex, numFolds,
                                                trainInputs, trainTargets, trainOutputs, trainCM,
                                                testInputs, testTargets, testOutputs, testCM,
                                                elapsedTime))
            {
                f64 trainElapsedTime = (f64)(sync::tTimer::usecTime() - trainStartTime);
                trainElapsedTime /= 1000000;  // usecs to secs
                trainObserver->didFinishTraining(learner, epochs, foldIndex, numFolds,
                                                 trainInputs, trainTargets, testInputs, testTargets,
                                                 trainElapsedTime);
                return epochs;
            }
        }
    }
}

u32  ezTrain(iLearner* learner,       std::vector< tIO >& trainInputs,
                                      std::vector< tIO >& trainTargets,
                                const std::vector< tIO >& testInputs,
                                const std::vector< tIO >& testTargets,
                                u32 batchSize,
                                iEZTrainObserver* trainObserver)
{
    return s_ezTrain(learner,
                     trainInputs, trainTargets,
                     testInputs, testTargets,
                     batchSize,
                     trainObserver,
                     0, 1);
}

u32  ezTrain(iLearner* learner, const std::vector< tIO >& allInputs,
                                const std::vector< tIO >& allTargets,
                                u32 batchSize, u32 numFolds,
                                iEZTrainObserver* trainObserver)
{
    if (allInputs.size() != allTargets.size())
        throw eInvalidArgument("The number of input and target vectors must be the same!");
    if (allInputs.size() == 0)
        throw eInvalidArgument("There must be at least one example.");
    if (numFolds == 0)
        throw eInvalidArgument("Zero folds makes no sense.");
    if (numFolds == 1)
        throw eInvalidArgument("One fold makes no sense.");

    f64 frac = 1.0 / numFolds;

    u32 accumEpochs = 0;

    for (u32 i = 0; i < numFolds; i++)
    {
        if (i > 0)
            learner->reset();

        // Calculate the range of the test set.
        u32 start = (u32) round((f64)allInputs.size() * frac*i);
        u32 end   = (u32) round((f64)allInputs.size() * frac*(i+1));

        // Build the training and test inputs.
        std::vector< tIO > trainInputs = allInputs;
        trainInputs.erase(trainInputs.begin()+start, trainInputs.begin()+end);
        std::vector< tIO > testInputs(allInputs.begin()+start, allInputs.begin()+end);

        // Build the training and test targets.
        std::vector< tIO > trainTargets = allTargets;
        trainTargets.erase(trainTargets.begin()+start, trainTargets.begin()+end);
        std::vector< tIO > testTargets(allTargets.begin()+start, allTargets.begin()+end);

        // Train!
        accumEpochs += s_ezTrain(learner,
                                 trainInputs, trainTargets,
                                 testInputs, testTargets,
                                 batchSize,
                                 trainObserver,
                                 i, numFolds);
    }

    return accumEpochs;
}

u32  ezTrain(iLearner* learner, const std::vector< tIO >& allInputs,
                                const std::vector< tIO >& allTargets,
                                u32 batchSize, std::vector<u32> foldSplitPoints,
                                iEZTrainObserver* trainObserver)
{
    if (allInputs.size() != allTargets.size())
        throw eInvalidArgument("The number of input and target vectors must be the same!");
    if (allInputs.size() == 0)
        throw eInvalidArgument("There must be at least one example.");
    u32 numFolds = (u32)foldSplitPoints.size() + 1;
    if (numFolds == 1)
        throw eInvalidArgument("One fold makes no sense.");

    u32 accumEpochs = 0;

    for (u32 i = 0; i < numFolds; i++)
    {
        if (i > 0)
            learner->reset();

        // Calculate the range of the test set.
        u32 start = (i == 0) ? 0 : foldSplitPoints[i-1];
        u32 end   = (i == numFolds-1) ? (u32)allInputs.size() : foldSplitPoints[i];

        // Build the training and test inputs.
        std::vector< tIO > trainInputs = allInputs;
        trainInputs.erase(trainInputs.begin()+start, trainInputs.begin()+end);
        std::vector< tIO > testInputs(allInputs.begin()+start, allInputs.begin()+end);

        // Build the training and test targets.
        std::vector< tIO > trainTargets = allTargets;
        trainTargets.erase(trainTargets.begin()+start, trainTargets.begin()+end);
        std::vector< tIO > testTargets(allTargets.begin()+start, allTargets.begin()+end);

        // Train!
        accumEpochs += s_ezTrain(learner,
                                 trainInputs, trainTargets,
                                 testInputs, testTargets,
                                 batchSize,
                                 trainObserver,
                                 i, numFolds);
    }

    return accumEpochs;
}


tSmartStoppingWrapper::tSmartStoppingWrapper(u32 minEpochs,
                                             u32 maxEpochs,
                                             f64 significantThreshold,
                                             f64 patienceIncrease,
                                             iEZTrainObserver* wrappedObserver,
                                             nPerformanceAttribute performanceAttribute)
    : m_minEpochs(minEpochs),
      m_maxEpochs(maxEpochs),
      m_significantThreshold(significantThreshold),
      m_patienceIncrease(patienceIncrease),
      m_obs(wrappedObserver),
      m_performanceAttribute(performanceAttribute)
{
    if (m_minEpochs == 0)
        throw eInvalidArgument("You must train for at least one epoch minimum.");
    if (m_maxEpochs < m_minEpochs)
        throw eInvalidArgument("max epochs must be >= min epochs");
    if (m_significantThreshold < 0.0)
        throw eInvalidArgument("The significance threshold cannot be less than zero.");
    if (m_significantThreshold >= 1.0)
        throw eInvalidArgument("The significance threshold must be less than 1.0.");
    if (m_patienceIncrease <= 1.0)
        throw eInvalidArgument("The patience increase must be greater than 1.0.");
    m_reset();
}

bool tSmartStoppingWrapper::didUpdate(iLearner* learner, const std::vector<tIO>& mostRecentBatch)
{
    return (!m_obs || m_obs->didUpdate(learner, mostRecentBatch));
}

bool tSmartStoppingWrapper::didFinishEpoch(iLearner* learner,
                                           u32 epochsCompleted,
                                           u32 foldIndex, u32 numFolds,
                                           const std::vector< tIO >& trainInputs,
                                           const std::vector< tIO >& trainTargets,
                                           const std::vector< tIO >& trainOutputs,
                                           const tConfusionMatrix& trainCM,
                                           const std::vector< tIO >& testInputs,
                                           const std::vector< tIO >& testTargets,
                                           const std::vector< tIO >& testOutputs,
                                           const tConfusionMatrix& testCM,
                                           f64 epochTrainTimeInSeconds)
{
    if (m_obs && !m_obs->didFinishEpoch(learner,
                                        epochsCompleted,
                                        foldIndex,
                                        numFolds,
                                        trainInputs,
                                        trainTargets,
                                        trainOutputs,
                                        trainCM,
                                        testInputs,
                                        testTargets,
                                        testOutputs,
                                        testCM,
                                        epochTrainTimeInSeconds))
    {
        return false;
    }

    f64 testError;
    switch (m_performanceAttribute)
    {
        case kClassificationErrorRate:
            testError = errorRate(testCM);
            break;
        case kOutputErrorMeasure:
            testError = learner->calculateError(testOutputs, testTargets);
            break;
        default:
            throw eLogicError("Unknown performance attribute");
    }
    if (testError <= m_bestTestErrorYet * (1.0 - m_significantThreshold))
    {
        m_bestTestErrorYet = testError;
        m_allowedEpochs = (u32)std::ceil(std::max((f64)m_minEpochs, epochsCompleted * m_patienceIncrease));
    }

    return (epochsCompleted < m_allowedEpochs && epochsCompleted < m_maxEpochs);
}

void tSmartStoppingWrapper::didFinishTraining(iLearner* learner,
                                              u32 epochsCompleted,
                                              u32 foldIndex, u32 numFolds,
                                              const std::vector< tIO >& trainInputs,
                                              const std::vector< tIO >& trainTargets,
                                              const std::vector< tIO >& testInputs,
                                              const std::vector< tIO >& testTargets,
                                              f64 trainingTimeInSeconds)
{
    if (m_obs) m_obs->didFinishTraining(learner, epochsCompleted, foldIndex, numFolds,
                                        trainInputs, trainTargets, testInputs, testTargets,
                                        trainingTimeInSeconds);
    m_reset();
}

void tSmartStoppingWrapper::m_reset()
{
    m_bestTestErrorYet = 1e100;
    m_allowedEpochs = m_minEpochs;
}


tBestRememberingWrapper::tBestRememberingWrapper(iEZTrainObserver* wrappedObserver,
                                                 nPerformanceAttribute performanceAttribute)
    : m_obs(wrappedObserver),
      m_performanceAttribute(performanceAttribute)
{
    reset();
}

void tBestRememberingWrapper::reset()
{
    m_bestTestEpochNum = 0;
    m_bestTestErrorRate = 1e100;
    m_bestTestOutputs.clear();
    m_bestTestCM.clear();
    m_matchingTrainCM.clear();
    m_serializedLearner.reset();
}

u32 tBestRememberingWrapper::bestTestEpochNum()  const
{
    return m_bestTestEpochNum;
}

f64 tBestRememberingWrapper::bestTestError() const
{
    return m_bestTestErrorRate;
}

const std::vector<tIO>& tBestRememberingWrapper::bestTestOutputs() const
{
    return m_bestTestOutputs;
}

const tConfusionMatrix& tBestRememberingWrapper::bestTestCM()      const
{
    return m_bestTestCM;
}

const tConfusionMatrix& tBestRememberingWrapper::matchingTrainCM() const
{
    return m_matchingTrainCM;
}

void tBestRememberingWrapper::newBestLearner(iLearner*& learner)   const
{
    tByteReadable readable(m_serializedLearner.getBuf());
    learner = iLearner::newDeserializedLearner(&readable);
}

bool tBestRememberingWrapper::didUpdate(iLearner* learner, const std::vector<tIO>& mostRecentBatch)
{
    return (!m_obs || m_obs->didUpdate(learner, mostRecentBatch));
}

bool tBestRememberingWrapper::didFinishEpoch(iLearner* learner,
                                             u32 epochsCompleted,
                                             u32 foldIndex, u32 numFolds,
                                             const std::vector< tIO >& trainInputs,
                                             const std::vector< tIO >& trainTargets,
                                             const std::vector< tIO >& trainOutputs,
                                             const tConfusionMatrix& trainCM,
                                             const std::vector< tIO >& testInputs,
                                             const std::vector< tIO >& testTargets,
                                             const std::vector< tIO >& testOutputs,
                                             const tConfusionMatrix& testCM,
                                             f64 epochTrainTimeInSeconds)
{
    // Delegate to the wrapped object whether or not to quit training.
    bool retVal = (!m_obs || m_obs->didFinishEpoch(learner,
                                                   epochsCompleted,
                                                   foldIndex,
                                                   numFolds,
                                                   trainInputs,
                                                   trainTargets,
                                                   trainOutputs,
                                                   trainCM,
                                                   testInputs,
                                                   testTargets,
                                                   testOutputs,
                                                   testCM,
                                                   epochTrainTimeInSeconds));

    // If this is the zero'th epoch, reset myself in case there was a fold that
    // happened before this point, in which case the state will still be for that
    // training sequence.
    if (epochsCompleted == 0)
        reset();

    // Evaluate the error rate on the test set and see if it's the best yet.
    f64 testErrorRate;
    switch (m_performanceAttribute)
    {
        case kClassificationErrorRate:
            testErrorRate = errorRate(testCM);
            break;
        case kOutputErrorMeasure:
            testErrorRate = learner->calculateError(testOutputs, testTargets);
            break;
        default:
            throw eLogicError("Unknown performance attribute");
    }
    if (testErrorRate < m_bestTestErrorRate)
    {
        m_bestTestEpochNum = epochsCompleted;
        m_bestTestErrorRate = testErrorRate;
        m_bestTestOutputs = testOutputs;
        m_bestTestCM = testCM;
        m_matchingTrainCM = trainCM;
        m_serializedLearner.reset();
        iLearner::serializeLearner(learner, &m_serializedLearner);
    }

    return retVal;
}

void tBestRememberingWrapper::didFinishTraining(iLearner* learner,
                                                u32 epochsCompleted,
                                                u32 foldIndex, u32 numFolds,
                                                const std::vector< tIO >& trainInputs,
                                                const std::vector< tIO >& trainTargets,
                                                const std::vector< tIO >& testInputs,
                                                const std::vector< tIO >& testTargets,
                                                f64 trainingTimeInSeconds)
{
    if (m_obs) m_obs->didFinishTraining(learner, epochsCompleted, foldIndex, numFolds,
                                        trainInputs, trainTargets, testInputs, testTargets,
                                        trainingTimeInSeconds);
}


tLoggingWrapper::tLoggingWrapper(u32 logInterval, bool isInputImageColor,
                                 u32 inputImageWidth, bool shouldDisplayAbsoluteValues,
                                 u32 cellWidthMultiplier,
                                 iEZTrainObserver* wrappedObserver,
                                 bool accumulateFoldIO,
                                 bool logVisuals,
                                 std::string fileprefix,
                                 nPerformanceAttribute performanceAttribute)
    : tBestRememberingWrapper(wrappedObserver, performanceAttribute),
      m_logInterval(logInterval),
      m_isColorInput(isInputImageColor),
      m_imageWidth(inputImageWidth),
      m_absoluteImage(shouldDisplayAbsoluteValues),
      m_cellWidthMultiplier(cellWidthMultiplier),
      m_accumulateFoldIO(accumulateFoldIO),
      m_logVisuals(logVisuals),
      m_fileprefix(fileprefix),
      m_performanceAttribute(performanceAttribute)
{
    if (m_logInterval == 0)
        throw eInvalidArgument("The log interval cannot be zero...");
}

tLoggingWrapper::~tLoggingWrapper()
{
    m_logfile.close();
    m_datafile.close();
}

bool tLoggingWrapper::didUpdate(iLearner* learner, const std::vector<tIO>& mostRecentBatch)
{
    // Nothing special to do here, so just call the super method.
    // The super method will call into the wrapped object.
    return tBestRememberingWrapper::didUpdate(learner, mostRecentBatch);
}

bool tLoggingWrapper::didFinishEpoch(iLearner* learner,
                                     u32 epochsCompleted,
                                     u32 foldIndex, u32 numFolds,
                                     const std::vector< tIO >& trainInputs,
                                     const std::vector< tIO >& trainTargets,
                                     const std::vector< tIO >& trainOutputs,
                                     const tConfusionMatrix& trainCM,
                                     const std::vector< tIO >& testInputs,
                                     const std::vector< tIO >& testTargets,
                                     const std::vector< tIO >& testOutputs,
                                     const tConfusionMatrix& testCM,
                                     f64 epochTrainTimeInSeconds)
{
    // Delegate to the super object whether or not to quit training.
    // The super method will call into the wrapped object.
    bool retVal = tBestRememberingWrapper::didFinishEpoch(learner,
                                                          epochsCompleted,
                                                          foldIndex,
                                                          numFolds,
                                                          trainInputs,
                                                          trainTargets,
                                                          trainOutputs,
                                                          trainCM,
                                                          testInputs,
                                                          testTargets,
                                                          testOutputs,
                                                          testCM,
                                                          epochTrainTimeInSeconds);

    // If this is the first callback, open the log files.
    if (epochsCompleted == 0 && foldIndex == 0)
    {
        m_logfile.open((m_fileprefix + learner->learnerInfoString() + ".log").c_str());
        m_datafile.open((m_fileprefix + learner->learnerInfoString() + ".data").c_str());
        learner->printLearnerInfo(m_logfile);
    }

    // Calculate error rates and output error measures for both
    // the training and test sets.
    f64 trainErrorRate = errorRate(trainCM);
    f64 testErrorRate  = errorRate(testCM);
    f64 trainError = learner->calculateError(trainOutputs, trainTargets);
    f64 testError = learner->calculateError(testOutputs, testTargets);

    // Print the training and test error to the human-readable log.
    m_logfile << "Train error:             " << trainErrorRate*100 << "% "
                                             << trainError << std::endl;
    m_logfile << "Test error:              " << testErrorRate*100 << "% "
                                             << testError << std::endl;
    m_logfile << std::endl;

    // Print the training and test error to the simplified data log.
    m_datafile << trainErrorRate*100 << " " << trainError << std::endl;
    m_datafile << testErrorRate*100 << " " << testError << std::endl;
    m_datafile << std::endl;

    // Save visuals every so many epochs.
    if ((epochsCompleted % m_logInterval) == 0)
    {
        std::ostringstream out;
        out << m_fileprefix << learner->learnerInfoString() << "__fold" << foldIndex+1 << "__epoch" << epochsCompleted;
        m_save(out.str(), learner, trainInputs, testInputs, testTargets, testOutputs);
    }

    return retVal;
}

static
void s_accumCM(tConfusionMatrix& accumCM, const tConfusionMatrix& newCM)
{
    if (accumCM.size() != newCM.size())
    {
        accumCM = newCM;
    }
    else
    {
        for (size_t i = 0; i < accumCM.size(); i++)
        {
            for (size_t j = 0; j < accumCM[i].size(); j++)
                accumCM[i][j] += newCM[i][j];
        }
    }
}

void tLoggingWrapper::didFinishTraining(iLearner* learner,
                                        u32 epochsCompleted,
                                        u32 foldIndex, u32 numFolds,
                                        const std::vector< tIO >& trainInputs,
                                        const std::vector< tIO >& trainTargets,
                                        const std::vector< tIO >& testInputs,
                                        const std::vector< tIO >& testTargets,
                                        f64 trainingTimeInSeconds)
{
    // The super method will call into the wrapped object.
    tBestRememberingWrapper::didFinishTraining(learner, epochsCompleted, foldIndex, numFolds,
                                               trainInputs, trainTargets, testInputs, testTargets,
                                               trainingTimeInSeconds);

    // Get a copy of the best found learner.
    iLearner* bestLearner = NULL;
    newBestLearner(bestLearner);

    // Accumulate the test set vectors and the CM from the best epoch if there will be
    // many folds.
    if (numFolds > 1 && m_accumulateFoldIO)
    {
        m_accumTestInputs.insert(m_accumTestInputs.end(), testInputs.begin(), testInputs.end());
        m_accumTestTargets.insert(m_accumTestTargets.end(), testTargets.begin(), testTargets.end());
        m_accumTestOutputs.insert(m_accumTestOutputs.end(), bestTestOutputs().begin(), bestTestOutputs().end());
        s_accumCM(m_accumTestCM, bestTestCM());
        s_accumCM(m_accumTrainCM, matchingTrainCM());
    }

    // Log the results of this fold.
    {
        switch (m_performanceAttribute)
        {
            case kClassificationErrorRate:
                m_logfile << "Best test classification error rate of " << bestTestError() * 100 << "% "
                          << "found after epoch " << bestTestEpochNum()
                          << "." << std::endl << std::endl;
                break;
            case kOutputErrorMeasure:
                m_logfile << "Best test output error measure of " << bestTestError() << " "
                          << "found after epoch " << bestTestEpochNum()
                          << "." << std::endl << std::endl;
                break;
            default:
                throw eLogicError("Unknown performance attribute");
        }
        m_logfile << "Training Set CM (fold=" << foldIndex+1 << '/' << numFolds << "):" << std::endl;
        print(matchingTrainCM(), m_logfile);
        m_logfile << "Test Set CM (fold=" << foldIndex+1 << '/' << numFolds << "):" << std::endl;
        print(bestTestCM(), m_logfile);
        std::ostringstream out;
        out << m_fileprefix << bestLearner->learnerInfoString() << "__fold" << foldIndex+1 << "__best";
        m_save(out.str(), bestLearner, trainInputs, testInputs, testTargets, bestTestOutputs());
    }

    // If this is the last of many folds, log the accumulated stuff.
    if (foldIndex+1 == numFolds && numFolds > 1 && m_accumulateFoldIO)
    {
        m_logfile << std::endl;
        m_logfile << "Accumulated Training Set CM:" << std::endl;
        print(m_accumTrainCM, m_logfile);
        m_logfile << "Accumulated Test Set CM:" << std::endl;
        print(m_accumTestCM, m_logfile);
        m_logfile << std::endl;

        m_logfile << "Num accumulated test examples: " << m_accumTestInputs.size() << std::endl;
        m_logfile << "Accumulated test classification error rate:   " << errorRate(m_accumTestCM)*100 << "%" << std::endl;
        m_logfile << std::endl;

        if (m_logVisuals)
        {
            img::tImage visualCM;
            buildVisualConfusionMatrix(m_accumTestInputs, m_isColorInput, m_imageWidth, m_absoluteImage,
                                       m_accumTestOutputs,
                                       m_accumTestTargets,
                                       &visualCM, m_cellWidthMultiplier);
            std::ostringstream out;
            out << m_fileprefix << bestLearner->learnerInfoString() << "__accum__cm.png";
            visualCM.saveToFile(out.str());
        }
    }

    // If this is the last fold (even if there was only one fold), save the outputs
    // of the learner. This is useful for doing post processing, such as for creating
    // ROC curves, or something like that.
    if (foldIndex+1 == numFolds && m_accumulateFoldIO)
    {
        const std::vector< tIO >& saveTheseTestInputs  = (numFolds > 1) ? m_accumTestInputs  : testInputs;
        const std::vector< tIO >& saveTheseTestTargets = (numFolds > 1) ? m_accumTestTargets : testTargets;
        const std::vector< tIO >& saveTheseTestOutputs = (numFolds > 1) ? m_accumTestOutputs : bestTestOutputs();
        tFileWritable outfile(m_fileprefix + bestLearner->learnerInfoString() + "__outputs.bin");
        rho::pack(&outfile, saveTheseTestInputs);
        rho::pack(&outfile, saveTheseTestTargets);
        rho::pack(&outfile, saveTheseTestOutputs);
    }

    // Delete the copy of the best learner.
    delete bestLearner;
}

void tLoggingWrapper::m_save(std::string filebasename,
                             iLearner* learner,
                             const std::vector<tIO>& trainInputs,
                             const std::vector<tIO>& testInputs,
                             const std::vector<tIO>& testTargets,
                             const std::vector<tIO>& testOutputs)
{
    // Save the learner.
    {
        tFileWritable file(filebasename + ".learner");
        iLearner::serializeLearner(learner, &file);
    }

    // Save a visual confusion matrix.
    if (m_logVisuals)
    {
        img::tImage visualCM;
        buildVisualConfusionMatrix(testInputs, m_isColorInput, m_imageWidth, m_absoluteImage,
                                   testOutputs,
                                   testTargets,
                                   &visualCM, m_cellWidthMultiplier);
        visualCM.saveToFile(filebasename + "__cm.png");
    }

    // Save a visual of the learner.
    if (m_logVisuals)
    {
        img::tImage image;
        visualize(learner, trainInputs.front(), m_isColorInput, m_imageWidth, m_absoluteImage, &image);
        image.saveToFile(filebasename + "__viz.png");
    }
}


}   // namespace ml
