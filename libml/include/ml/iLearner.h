#ifndef __ml_iLearner_h__
#define __ml_iLearner_h__


#include <ml/rhocompat.h>

#include <rho/types.h>
#include <rho/iPackable.h>

#include <vector>


namespace ml
{


/**
 * We can easily switch floating point precisions with the
 * following typedef. This effect the precision of the
 * input/output/target examples and the internal precision
 * of the learners.
 */
typedef f64 fml;


/**
 * This object will be used to denote input to the learner as
 * well as output from the learner. Both input/output to/from
 * this learner are vector data.
 *
 * This typedef makes it easier to represent several input examples
 * or several target examples without having to explicitly declare
 * vectors of vectors of floats.
 */
typedef std::vector<fml> tIO;


/**
 * This learner learns with vector data input and vector data output.
 */
class iLearner
{
    public:

        /**
         * Shows the learner one example. The learner will calculate error rates
         * (or whatever it does with examples), then accumulate the error in some
         * way or another.
         *
         * The learner will not become smarter by calling this method. You must
         * subsequently call update().
         */
        virtual void addExample(const tIO& input, const tIO& target) = 0;

        /**
         * Updates the learner to account for all the examples it's seen since
         * the last call to update().
         *
         * The accumulated error rates (or whatever) are then cleared.
         */
        virtual void update() = 0;

        /**
         * Uses the current knowledge of the learner to evaluate the given input.
         */
        virtual void evaluate(const tIO& input, tIO& output) const = 0;

        /**
         * Uses the current knowledge of the learner to evaluate the given inputs.
         *
         * This is the same as the above version of evaluate(), but this one
         * does a batch-evaluate, which is more efficient for most learners
         * to perform.
         */
        virtual void evaluateBatch(const std::vector<tIO>& inputs,
                                         std::vector<tIO>& outputs) const = 0;

        /**
         * Uses the current knowledge of the learner to evaluate the given inputs.
         *
         * This is the same as the above version of evaluateBatch(), but this one
         * takes iterators so that you can avoid copying data in some cases.
         */
        virtual void evaluateBatch(std::vector<tIO>::const_iterator inputStart,
                                   std::vector<tIO>::const_iterator inputEnd,
                                   std::vector<tIO>::iterator outputStart) const = 0;

        /**
         * Asks the learner to calculate the error between the given output
         * and the given target. For example, the learner may calculate
         * the standard squared error or the cross-entropy loss, if one of
         * those is appropriate. Or the learner may do something else.
         */
        virtual f64 calculateError(const tIO& output, const tIO& target) = 0;

        /**
         * Asks the learner to calculate the error between all the given
         * output/target pairs. For example, the learner may calculate
         * the average standard squared error or the average cross-entropy
         * loss, if one of those is appropriate. Or the learner may do
         * something else.
         */
        virtual f64 calculateError(const std::vector<tIO>& outputs,
                                   const std::vector<tIO>& targets) = 0;

        /**
         * Resets the learner to its initial state.
         */
        virtual void reset() = 0;

        /**
         * Prints the learner's configuration in a readable format.
         */
        virtual void printLearnerInfo(std::ostream& out) const = 0;

        /**
         * Returns a single-line version of printLearnerInfo().
         */
        virtual std::string learnerInfoString() const = 0;

        /**
         * Virtual dtor...
         */
        virtual ~iLearner() { }

    public:

        /**
         * Deserialize from the input stream a new copy of an iLearner
         * object.
         *
         * The caller of this method must call delete on the returned
         * value when they are finished using it.
         */
        static iLearner* newDeserializedLearner(iReadable* readable);

        /**
         * Serializes a learner to the given output stream.
         */
        static void serializeLearner(iLearner* learner, iWritable* writable);
};


}   // namespace ml


#endif  // __ml_iLearner_h__
