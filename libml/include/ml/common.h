#ifndef __ml_common_h__
#define __ml_common_h__


#include <ml/rhocompat.h>
#include <ml/iLearner.h>

#include <rho/eRho.h>
#include <rho/types.h>
#include <rho/img/tImage.h>
#include <rho/algo/stat_util.h>

#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>


namespace ml
{


//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
// The logistic function:
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

f64 logistic_function(f64 z);
f64 derivative_of_logistic_function(f64 z);
f64 inverse_of_logistic_function(f64 y);
f64 logistic_function_min();
f64 logistic_function_max();


//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
// The hyperbolic function:
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

f64 hyperbolic_function(f64 z);
f64 derivative_of_hyperbolic_function(f64 z);
f64 inverse_of_hyperbolic_function(f64 y);
f64 hyperbolic_function_min();
f64 hyperbolic_function_max();


//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
// IO manipulation tools:
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

/**
 * Turns the integer value into an example that can be trained-on.
 * The returned training example has 'numDimensions' number of
 * dimensions, where one dimension is set to 1.0, and all others
 * are set to 0.0. The high dimension's index is given by
 * 'highDimension'.
 *
 * This is useful for creating the target vector when training
 * a classifier.
 */
tIO examplify(u32 highDimension, u32 numDimensions);

/**
 * Does the opposite operation as the above examplify() function.
 * It does so by determining which dimension has the highest
 * value, and returns the index to that dimension.
 *
 * If 'error' is not NULL, the std squared error between the given
 * output and the "correct" output is calculated and stored in 'error'.
 * The "correct" output is obtained by calling the examplify()
 * function above. The assumption is made that the returned
 * index for the highest dimension is correct, thus the method
 * calculates the standard error between the given output and
 * the "correct" output.
 *
 * This is useful for evaluating the output of a classifier.
 */
u32 un_examplify(const tIO& output, f64* error = NULL);

/**
 * Turns the given image into an example that can be trained-on.
 */
tIO examplify(const img::tImage* image);

/**
 * Generates an image representation of the given tIO object, 'io'.
 *
 * If 'io' should be interpreted as an RGB image, set 'color'
 * to true. If the 'io' should be interpreted as a grey image,
 * set 'color' to false.
 *
 * You must specify the 'width' of the generated image. The
 * height will be derived by this function.
 *
 * If 'color' is false, then the green channel of the output image
 * is used to indicate positive values in 'io' and the red channel
 * of the output image is used to indicate negative values in 'io'.
 *
 * If 'color' is true, the trick above cannot be used because we
 * need each channel of the output image to represent itself. In
 * this case, the 'absolute' parameter is used to help determine
 * how to generate the output image.
 *
 * If 'absolute' is set to true, the absolute value of 'io'
 * will be used when producing the image. Otherwise, the relative
 * values will be used to produce the image (meaning that values
 * equal to zero will not be black if there are any negative values
 * in 'io').
 *
 * If the data has a finite range, you can specify that range
 * so that un_examplify() can create an image that respects
 * it. Otherwise, un_examplify() will use the min and max
 * of the data itself as the range so that the generated
 * image uses the full range of color.
 *
 * The generated image is stored in 'dest'.
 */
void un_examplify(const tIO& io, bool color, u32 width,
                  bool absolute, img::tImage* dest,
                  const f64* minValue = NULL, const f64* maxValue = NULL);

/**
 * Z-score a set of input examples.
 *
 * Z-scoring is transforming the data so that each dimension's mean is zero
 * and its standard deviation is one.
 *
 * The 'dStart' and 'dEnd' indices let you specify which dimensions of the
 * data you are interesting in zscoring. The dimensions that will be zscored
 * are [dStart, dEnd). Note that by default all dimensions will be zscored.
 */
void zscore(std::vector<tIO>& inputs, u32 dStart=0, u32 dEnd=0xFFFFFFFF);

/**
 * Z-score the training set, and z-score the test set to match.
 *
 * Z-scoring is transforming the data so that each dimension's mean is zero
 * and its standard deviation is one.
 */
void zscore(std::vector<tIO>& trainingInputs,
            std::vector<tIO>& testInputs);


//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
// Error measures:
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

/**
 * Calculates and returns the standard squared error between the given
 * output and the given target.
 */
f64 standardSquaredError(const tIO& output, const tIO& target);

/**
 * Calculates the average standard squared error between each output/target
 * pair.
 */
f64 standardSquaredError(const std::vector<tIO>& outputs,
                         const std::vector<tIO>& targets);

/**
 * Calculates and returns the cross-entropy cost between the given
 * output and the given target.
 */
f64 crossEntropyCost(const tIO& output, const tIO& target);

/**
 * Calculates the average cross-entropy cost between each output/target
 * pair.
 */
f64 crossEntropyCost(const std::vector<tIO>& outputs,
                     const std::vector<tIO>& targets);

/**
 * Calculates the root-mean-squared error of the output/target
 * pairs.
 *
 * (Note: This error function does not have a single output/target
 *        pair version because that wouldn't not make sense for this
 *        particular way of measuring error.)
 */
f64 rmsError(const std::vector<tIO>& outputs,
             const std::vector<tIO>& targets);


//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
// Confusion matrix tools:
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

/**
 * This typedef makes creating a confusion matrix easier.
 */
typedef std::vector< std::vector<u32> > tConfusionMatrix;

/**
 * Creates a confusion matrix for the given output/target
 * pairs. For each output/target pair, un_examplify() is
 * called twice (once for the output and once for the target),
 * and the corresponding entry in the confusion matrix is
 * incremented.
 */
void buildConfusionMatrix(const std::vector<tIO>& outputs,
                          const std::vector<tIO>& targets,
                                tConfusionMatrix& confusionMatrix);

/**
 * Same as buildConfusionMatrix() above, but this function
 * does not simply count the entries in each cell of the
 * confusion matrix, it actually draws the input examples in
 * the cells of the confusion matrix! This gives you a
 * visual representation of the confusion matrix.
 *
 * The inputs are assumed to be images, for how else could
 * we draw them!? The inputs are transformed into images
 * by calling un_examplify() on them. See the comments
 * of un_examplify() for details on 'color', 'width', and
 * 'absolute'.
 *
 * The resulting image is stored in 'dest'.
 *
 * The width of each cell is calculated by multiplying the
 * 'width' by 'cellWidthMultiplier'. So the width of 'dest'
 * will be 'width*cellWidthMultiplier*<num_classes>'
 */
void buildVisualConfusionMatrix(const std::vector<tIO>& inputs,
                                bool color, u32 width, bool absolute,
                                const std::vector<tIO>& outputs,
                                const std::vector<tIO>& targets,
                                      img::tImage* dest,
                                u32 cellWidthMultiplier = 5);

/**
 * Prints the confusion matrix in a pretty format.
 */
void print(const tConfusionMatrix& confusionMatrix, std::ostream& out);

/**
 * Calculates the error rate for the given confusion matrix.
 *
 * Works for any confusion matrix.
 */
f64  errorRate(const tConfusionMatrix& confusionMatrix);

/**
 * Calculates the accuracy for the given confusion matrix.
 *
 * Works for any confusion matrix.
 */
f64  accuracy(const tConfusionMatrix& confusionMatrix);

/**
 * Calculates the precision of the confusion matrix.
 *
 * Only works for confusion matrices that have true/false
 * dimensions (aka, confusion matrices that are two-by-two).
 */
f64  precision(const tConfusionMatrix& confusionMatrix);

/**
 * Calculates the recall of the confusion matrix.
 *
 * Only works for confusion matrices that have true/false
 * dimensions (aka, confusion matrices that are two-by-two).
 */
f64  recall(const tConfusionMatrix& confusionMatrix);


//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
// Training and visualization low-level helpers:
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

class iTrainObserver
{
    public:

        /**
         * This method is called by the train() function below after
         * update() has been called on the given learner. It should
         * return true if all is well and training should continue.
         * It should return false if the training process should
         * halt. This is useful if you need to cancel training
         * due to user input, or something like that.
         */
        virtual bool didUpdate(iLearner* learner, const std::vector<tIO>& mostRecentBatch) = 0;

        virtual ~iTrainObserver() { }
};

/**
 * This function trains the leaner on the given examples,
 * calling the training observer after each batch has been
 * processed by the learner. This function returns true if
 * the training process completed fully, and it returns false
 * if the training observer indicated that training should
 * halt early.
 *
 * Use this for training on the training-set.
 */
bool train(iLearner* learner, const std::vector<tIO>& inputs,
                              const std::vector<tIO>& targets,
                              u32 batchSize,
                              iTrainObserver* trainObserver = NULL);

/**
 * This function tests the learner on the given examples.
 * The 'outputs' parameter is populated by this function,
 * making it easy to create a confusion matrix after this
 * function returns (assuming, of course, that you have
 * the targets for each of the inputs given to this function).
 *
 * Use this for evaluating how the learner is doing on the
 * training-set AND test-set. Know how the learner is doing
 * on each is vital.
 */
void evaluate(iLearner* learner, const std::vector<tIO>& inputs,
                                       std::vector<tIO>& outputs,
                                 u32 batchSize=100);

/**
 * Creates a visual of the learner processing the example provided.
 * The visual is stored as an image in 'dest'.
 */
void visualize(iLearner* learner, const tIO& example,
               bool color, u32 width, bool absolute,
               img::tImage* dest);


//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
// Training medium-level helpers:
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

class iEZTrainObserver : public iTrainObserver
{
    public:

        /**
         * This method is called by both ezTrain() functions below after
         * a full epoch of training has been done on the given learner.
         * It should return true if all is well and training should continue.
         * It should return false if the training process should halt. This
         * is useful if you need to cancel training due to user input, or
         * if you detect that the learner has been trained enough and is ready
         * to be used.
         */
        virtual bool didFinishEpoch(iLearner* learner,
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
                                    f64 epochTrainTimeInSeconds) = 0;

        /**
         * This method is called after training completes, meaning that
         * didFinishEpoch() will not be called anymore for this foldIndex.
         */
        virtual void didFinishTraining(iLearner* learner,
                                       u32 epochsCompleted,
                                       u32 foldIndex, u32 numFolds,
                                       const std::vector< tIO >& trainInputs,
                                       const std::vector< tIO >& trainTargets,
                                       const std::vector< tIO >& testInputs,
                                       const std::vector< tIO >& testTargets,
                                       f64 trainingTimeInSeconds) = 0;
};

/**
 * This function trains the leaner on the given training set,
 * and tests the learner on the given test set. It trains
 * for as many epochs are needed by calling the train()
 * function above to train the learner on each epoch. This
 * function takes a train observer which it notifies (if not
 * null) after each epoch with the most recent training results.
 * This function will always pass foldIndex=0 and numFolds=1 to
 * the train observer. This function will not return until the
 * observer indicates that training can halt.
 *
 * This function returns the number of epochs of training which
 * were completed.
 *
 * This function is intended to replace calling train() in most
 * application where straight-forward training is needed.
 */
u32  ezTrain(iLearner* learner,       std::vector< tIO >& trainInputs,
                                      std::vector< tIO >& trainTargets,
                                const std::vector< tIO >& testInputs,
                                const std::vector< tIO >& testTargets,
                                u32 batchSize,
                                iEZTrainObserver* trainObserver = NULL);

/**
 * This function is a lot like the ezTrain() function above,
 * but this function is used when you do not have a dedicated
 * testing set, meaning you need to do something like ten-fold
 * cross-validation.
 *
 * This function behaves exactly like ezTrain() above, but it
 * trains the learner fresh over-and-over with a different test
 * set (aka "hold-out set") on each iteration. You should use
 * the train observer to accumulate the hold-out error after
 * each fold iteration so that you have a complete idea of the
 * learner's generalization error.
 *
 * This function sets foldIndex and numFolds appropriately to the
 * train observer. It will always set numFolds to the observer
 * equal to numFolds passed into this function, and it will set
 * foldIndex equal to the index of the current fold (zero-indexed).
 *
 * This function returns the number of epochs of training which
 * were completed (accumulated over all the training folds).
 *
 * Like the above ezTrain() function, this function is intended to
 * replace calling train() in most application where straight-forward
 * x-fold cross-validation training is needed.
 */
u32  ezTrain(iLearner* learner, const std::vector< tIO >& allInputs,
                                const std::vector< tIO >& allTargets,
                                u32 batchSize, u32 numFolds,
                                iEZTrainObserver* trainObserver = NULL);

/**
 * This is also a version of ezTrain() for doing x-fold cross-validation,
 * but this version is used when you want to tell the function where
 * to split the dataset for each fold.
 *
 * This is necessary when you have created "extra" training data by
 * duplicating examples (with mutation). For example, you may have scaled
 * and/or translated example images to create more examples and inject
 * prior-knowledge into the system.
 * The results would be misleading if you let ezTrain() split such a dataset
 * anywhere it wanted, for you don't want to train with some data and test
 * on data which is a duplication of the training data. Instead, you should
 * tell ezTrain() where to split the dataset so that each duplicated example is
 * entirely in the training portion or the test portion, but does not overlap
 * the two.
 *
 * In this function, the number of training folds will equal:
 *          foldSplitPoints.size() + 1
 *
 * Each entry in foldSplitPoints indicates the first index of a fold.
 * Do not specify index-0 in your foldSplitPoints array. It is assumed
 * that the first fold will begin at index 0 and continue until the
 * index found in foldSplitPoints[0].
 */
u32  ezTrain(iLearner* learner, const std::vector< tIO >& allInputs,
                                const std::vector< tIO >& allTargets,
                                u32 batchSize, std::vector<u32> foldSplitPoints,
                                iEZTrainObserver* trainObserver = NULL);


//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////
// Training high-level helpers:
//////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////

enum nPerformanceAttribute
{
    kClassificationErrorRate      = 1,
    kOutputErrorMeasure           = 2
};

class tSmartStoppingWrapper : public iEZTrainObserver
{
    public:

        /**
         * This class defines an early stopping condition for training
         * a learner.
         *
         * It guarantees that the learner will be trained for at least
         * 'minEpochs' number of epochs, even if no progress is seen.
         *
         * It guarantees that the learner will be trained for at most
         * 'maxEpochs' number of epochs, even if progress is seen the
         * entire time.
         *
         * The algorithm respects significant improvements in performance
         * and may increase the allowed training time when a significant
         * improvement is encountered. If performance increases by
         * 'significantThreshold' or more, the increase is considered
         * significant. When a significant improvement happens, the allowed
         * training time is extended to be at least the current duration of
         * training time multiplied by 'patienceIncrease'.
         *
         * This class wraps another iEZTrainObserver so that observers
         * can be decorated by objects of this type.
         *
         * You can choose which performance measure is used to determine
         * when to stop with the last parameter.
         */
        tSmartStoppingWrapper(u32 minEpochs=50,
                              u32 maxEpochs=1000,
                              f64 significantThreshold=0.005,   // <-- half a percent
                              f64 patienceIncrease=2.0,
                              iEZTrainObserver* wrappedObserver=NULL,
                              nPerformanceAttribute performanceAttribute=kOutputErrorMeasure);

    public:

        // iTrainObserver interface:
        bool didUpdate(iLearner* learner, const std::vector<tIO>& mostRecentBatch);

        // iEZTrainObserver interface:
        bool didFinishEpoch(iLearner* learner,
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
                            f64 epochTrainTimeInSeconds);
        void didFinishTraining(iLearner* learner,
                               u32 epochsCompleted,
                               u32 foldIndex, u32 numFolds,
                               const std::vector< tIO >& trainInputs,
                               const std::vector< tIO >& trainTargets,
                               const std::vector< tIO >& testInputs,
                               const std::vector< tIO >& testTargets,
                               f64 trainingTimeInSeconds);

    private:

        void m_reset();

    private:

        const u32 m_minEpochs;
        const u32 m_maxEpochs;
        const f64 m_significantThreshold;
        const f64 m_patienceIncrease;

        iEZTrainObserver * const m_obs;

        f64 m_bestTestErrorYet;
        u32 m_allowedEpochs;

        nPerformanceAttribute m_performanceAttribute;
};


class tBestRememberingWrapper : public iEZTrainObserver
{
    public:

        /**
         * This class wraps a iEZTrainObserver to add the ability to
         * remember the learner and the output values which performed
         * best on the test set during training. This allows you to identify
         * which point in the learning process gave you the best generalization
         * error estimate, and then to report that point at the end of training.
         *
         * Using 'performanceAttribute', you can choose which performance measure
         * is used for determining the best network observed.
         */
        tBestRememberingWrapper(iEZTrainObserver* wrappedObserver=NULL,
                                nPerformanceAttribute performanceAttribute=kClassificationErrorRate);

        void reset();

        u32 bestTestEpochNum()  const;
        f64 bestTestError() const;

        const std::vector<tIO>& bestTestOutputs() const;
        const tConfusionMatrix& bestTestCM()      const;
        const tConfusionMatrix& matchingTrainCM() const;

        void newBestLearner(iLearner*& learner)   const;   // <-- caller must "delete learner;" when finished with it

    public:

        // iTrainObserver interface:
        bool didUpdate(iLearner* learner, const std::vector<tIO>& mostRecentBatch);

        // iEZTrainObserver interface:
        bool didFinishEpoch(iLearner* learner,
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
                            f64 epochTrainTimeInSeconds);
        void didFinishTraining(iLearner* learner,
                               u32 epochsCompleted,
                               u32 foldIndex, u32 numFolds,
                               const std::vector< tIO >& trainInputs,
                               const std::vector< tIO >& trainTargets,
                               const std::vector< tIO >& testInputs,
                               const std::vector< tIO >& testTargets,
                               f64 trainingTimeInSeconds);

    private:

        u32 m_bestTestEpochNum;
        f64 m_bestTestErrorRate;

        std::vector<tIO> m_bestTestOutputs;
        tConfusionMatrix m_bestTestCM;
        tConfusionMatrix m_matchingTrainCM;

        tByteWritable m_serializedLearner;

        iEZTrainObserver * const m_obs;

        nPerformanceAttribute m_performanceAttribute;
};


class tLoggingWrapper : public tBestRememberingWrapper
{
    public:

        /**
         * This class provides a wrapper around another iEZTrainObserver
         * to decorate it with logging ability. Note that this class
         * extends tBestRememberingWrapper, so if you need the functionality
         * of tBestRememberingWrapper you do not need to add it to the
         * decoration chain yourself because it comes free when you use this
         * class.
         *
         * Error rates are logged every epoch to a human-readable log file
         * and to a simplified data log file. Every log file produced by
         * this class is prefixed with the 'fileprefix' string specified
         * to the constructor.
         *
         * Every 'logInterval' number of epochs, the learner itself is
         * serialized to a file, and visualization of the learner and its
         * progress are also saved if 'logVisuals' is true.
         *
         * See un_examplify() for a description of 'isInputImageColor',
         * 'inputImageWidth', and 'shouldDisplayAbsoluteValues'. These
         * parameters are needed for creating the visualization which occur
         * every 'logInterval' number of epochs. These parameters are only
         * relevant if 'logVisuals' is true.
         *
         * See buildConfusionMatrix() for a description of 'cellWidthMultiplier'.
         * This parameter is only relevant if 'logVisuals' is true.
         *
         * The visualizations of the learner are useful when the learner is
         * processing image data. If the input data is not image data,
         * the visuals are not as meaningful, and in some cases cannot be
         * produced anyway because of data alignment issues. You use
         * 'logVisuals' to turn off the logging of these visualizations.
         *
         * See tBestRememberingWrapper for a description of 'performanceAttribute'.
         *
         * This logging wrapper works well when used with both versions of
         * ezTrain() above. That is, it works for normal train/test sets,
         * and it works for x-fold cross-validation sets.
         *
         * If 'accumulateFoldIO' is set, a few things will happen:
         *     1. If using more than one fold, the test inputs/targets/outputs
         *        of each fold will be accumulated and will be displayed
         *        at the end of learning (such as the accumulated CM and
         *        the accumulated error rate and an accumulated visual CM).
         *        Note: Accumulating the test inputs/targets/outputs like
         *        this is memory intensive when the training set is large.
         *     2. A binary log file will be written that contains
         *        the accumulated test inputs/targets/outputs for each fold
         *        (this will happen even if there is only one fold). This
         *        binary file is useful for doing post processing on the
         *        outputs, such as evaluating the outputs using a different
         *        cost function, or trying to combine or morph the outputs
         *        to decrease the error in some way.
         *        Note: This binary log file will be very large if the training
         *        set is large.
         */
        tLoggingWrapper(u32 logInterval, bool isInputImageColor,
                        u32 inputImageWidth, bool shouldDisplayAbsoluteValues,
                        u32 cellWidthMultiplier = 5,
                        iEZTrainObserver* wrappedObserver = NULL,
                        bool accumulateFoldIO = false,
                        bool logVisuals = true,
                        std::string fileprefix=std::string(),
                        nPerformanceAttribute performanceAttribute=kClassificationErrorRate);

        ~tLoggingWrapper();

    public:

        // iTrainObserver interface:
        bool didUpdate(iLearner* learner, const std::vector<tIO>& mostRecentBatch);

        // iEZTrainObserver interface:
        bool didFinishEpoch(iLearner* learner,
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
                            f64 epochTrainTimeInSeconds);
        void didFinishTraining(iLearner* learner,
                               u32 epochsCompleted,
                               u32 foldIndex, u32 numFolds,
                               const std::vector< tIO >& trainInputs,
                               const std::vector< tIO >& trainTargets,
                               const std::vector< tIO >& testInputs,
                               const std::vector< tIO >& testTargets,
                               f64 trainingTimeInSeconds);

    private:

        void m_save(std::string filebasename,
                    iLearner* learner,
                    const std::vector<tIO>& trainInputs,
                    const std::vector<tIO>& testInputs,
                    const std::vector<tIO>& testTargets,
                    const std::vector<tIO>& testOutputs);

    private:

        const u32  m_logInterval;
        const bool m_isColorInput;
        const u32  m_imageWidth;
        const bool m_absoluteImage;
        const u32  m_cellWidthMultiplier;

        const bool m_accumulateFoldIO;
        const bool m_logVisuals;

        std::string m_fileprefix;

        std::ofstream m_logfile;
        std::ofstream m_datafile;

        std::vector<tIO> m_accumTestInputs;     //
        std::vector<tIO> m_accumTestTargets;    //
        std::vector<tIO> m_accumTestOutputs;    //  These are only used if m_accumulateFoldIO is true
        tConfusionMatrix m_accumTestCM;         //
        tConfusionMatrix m_accumTrainCM;        //

        nPerformanceAttribute m_performanceAttribute;
};


}    // namespace ml


#endif   // __ml_common_h__
