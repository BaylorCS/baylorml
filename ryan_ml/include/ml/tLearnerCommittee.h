#ifndef __ml_tLearnerCommittee_h__
#define __ml_tLearnerCommittee_h__


#include <ml/rhocompat.h>
#include <ml/common.h>
#include <ml/iLearner.h>

#include <rho/refc.h>
#include <rho/sync/tThreadPool.h>

#include <vector>


namespace ml
{


class tLearnerCommittee : public iLearner, public bNonCopyable
{
    public:

        /**
         * An enum for the types of scoring within the committee that will
         * be used.
         */
        enum nCommitteeType
        {
            kCommitteeAverage,           // average the outputs of each learner
            kCommitteeMostConfident,     // trust the learner with the largest output value (largest confidence)
            kCommitteeTypeMaxValue       // marks the end of this enum; do not use this
        };

        /**
         * Constructs a committee of trained learners.
         */
        tLearnerCommittee(const std::vector< refc<iLearner> >& committee,
                          nCommitteeType type=kCommitteeAverage);

        /**
         * Constructs a committee of trained learners, and uses a thread
         * pool of the given size for evalutate().
         */
        tLearnerCommittee(const std::vector< refc<iLearner> >& committee,
                          u32 threadPoolSize,
                          nCommitteeType type=kCommitteeAverage);

        /**
         * D'tor
         */
        ~tLearnerCommittee();

        /////////////////////////////////////////////////////////////////////
        // iLearner interface
        /////////////////////////////////////////////////////////////////////

        /**
         * Evaluate the input using the committee.
         */
        void evaluate(const tIO& input, tIO& output) const;

        /**
         * Evaluate the input using the committee.
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
         * Prints the each of the committee's learner's configuration.
         */
        void printLearnerInfo(std::ostream& out) const;

        /**
         * Returns a single-line version of printLearnerInfo().
         */
        std::string learnerInfoString() const;

        /**
         * Calculates the average error of the learners in the committee.
         */
        f64 calculateError(const tIO& output, const tIO& target);

        /**
         * Calculates the average error of the learners in the committee.
         */
        f64 calculateError(const std::vector<tIO>& outputs,
                           const std::vector<tIO>& targets);

        /**
         * Do not call this. You cannot train a committee. They must be
         * already trained...
         */
        void addExample(const tIO& input, const tIO& target);

        /**
         * Do not call this. You cannot train a committee. They must be
         * already trained...
         */
        void update();

        /**
         * Do not call this. Resetting the committee makes no sense
         * because the committee cannot be re-trained after the reset.
         */
        void reset();

    private:

        std::vector< refc<iLearner> > m_committee;
        nCommitteeType m_type;

        sync::tThreadPool* m_threadPool;
};


}   // namespace ml


#endif   // __ml_tLearnerCommittee_h__
