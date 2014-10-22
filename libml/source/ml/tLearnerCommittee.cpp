#include <ml/tLearnerCommittee.h>


namespace ml
{


tLearnerCommittee::tLearnerCommittee(const std::vector< refc<iLearner> >& committee, nCommitteeType type)
    : m_committee(committee),
      m_type(type),
      m_threadPool(NULL)
{
    if (m_committee.size() == 0)
        throw eInvalidArgument("A committee must be one or more learners.");
    if (m_type < 0 || m_type >= kCommitteeTypeMaxValue)
        throw eInvalidArgument("Invalid committee type enum value.");
}

tLearnerCommittee::tLearnerCommittee(const std::vector< refc<iLearner> >& committee,
                                     u32 threadPoolSize, nCommitteeType type)
    : m_committee(committee),
      m_type(type),
      m_threadPool(NULL)
{
    if (m_committee.size() == 0)
        throw eInvalidArgument("A committee must be one or more learners.");
    if (m_type < 0 || m_type >= kCommitteeTypeMaxValue)
        throw eInvalidArgument("Invalid committee type enum value.");
    m_threadPool = new sync::tThreadPool(threadPoolSize);
}

tLearnerCommittee::~tLearnerCommittee()
{
    delete m_threadPool;
    m_threadPool = NULL;
}

static
void s_accum(tIO& accum, const tIO& out)
{
    if (accum.size() != out.size())
        throw eLogicError("The learners of a committee must have the same output dimensionality.");
    for (size_t i = 0; i < accum.size(); i++)
        accum[i] += out[i];
}

template <class T>
T s_max(const std::vector<T>& vect)
{
    if (vect.size() == 0)
        throw eLogicError("To calculate the max of a vector it cannot be zero-length.");
    T ma = vect[0];
    for (size_t i = 1; i < vect.size(); i++)
        ma = std::max(ma, vect[i]);
    return ma;
}

class tEvalWorker : public sync::iRunnable, public bNonCopyable
{
    public:

        tEvalWorker(const iLearner* learner,
                    std::vector<tIO>::const_iterator inputStart,
                    std::vector<tIO>::const_iterator inputEnd)
            : m_learner(learner),
              m_inputStart(inputStart),
              m_inputEnd(inputEnd)
        {
        }

        void run()
        {
            m_output.resize(m_inputEnd-m_inputStart);
            m_learner->evaluateBatch(m_inputStart, m_inputEnd,
                                     m_output.begin());
        }

        const std::vector<tIO>& getOutput() const
        {
            return m_output;
        }

    private:

        const iLearner* m_learner;

        std::vector<tIO>::const_iterator m_inputStart;
        std::vector<tIO>::const_iterator m_inputEnd;

        std::vector<tIO> m_output;
};

void tLearnerCommittee::evaluate(const tIO& input, tIO& output) const
{
    std::vector<tIO> inputs; inputs.push_back(input);
    std::vector<tIO> outputs;
    evaluateBatch(inputs, outputs);
    output = outputs[0];
}

void tLearnerCommittee::evaluateBatch(const std::vector<tIO>& inputs,
                                            std::vector<tIO>& outputs) const
{
    outputs.resize(inputs.size());
    evaluateBatch(inputs.begin(), inputs.end(), outputs.begin());
}

void tLearnerCommittee::evaluateBatch(std::vector<tIO>::const_iterator inputStart,
                                      std::vector<tIO>::const_iterator inputEnd,
                                      std::vector<tIO>::iterator outputStart) const
{
    if ((inputEnd-inputStart) <= 0)
        throw eInvalidArgument("The batch size cannot be zero.");

    std::vector< std::vector<tIO> > outputs;
    if (m_threadPool)
    {
        std::vector< refc<sync::iRunnable> > runnables;
        std::vector<sync::tThreadPool::tTaskKey> taskKeys;
        for (size_t i = 0; i < m_committee.size(); i++)
        {
            refc<sync::iRunnable> runnable(new tEvalWorker(m_committee[i], inputStart, inputEnd));
            taskKeys.push_back(m_threadPool->push(runnable));
            runnables.push_back(runnable);
        }
        for (size_t i = 0; i < taskKeys.size(); i++)
        {
            m_threadPool->wait(taskKeys[i]);
        }
        for (size_t i = 0; i < runnables.size(); i++)
        {
            sync::iRunnable* runnable = runnables[i];
            tEvalWorker* worker = dynamic_cast<tEvalWorker*>(runnable);
            outputs.push_back(worker->getOutput());
        }
    }
    else
    {
        for (size_t i = 0; i < m_committee.size(); i++)
        {
            std::vector<tIO> outHere(inputEnd-inputStart);
            m_committee[i]->evaluateBatch(inputStart, inputEnd, outHere.begin());
            outputs.push_back(outHere);
        }
    }

    if (outputs.size() != m_committee.size() || outputs.size() == 0)
        throw eLogicError("For some reason, we didn't get the output from every learner in this committee.");
    for (size_t i = 1; i < outputs.size(); i++)
        if (outputs[i].size() != outputs[0].size() || outputs[i].size() == 0)
            throw eLogicError("The batch size of each learner in this committee does not match.");
    size_t batchSize = outputs[0].size();
    if (batchSize != ((size_t)(inputEnd-inputStart)))
        throw eLogicError("The batch size of the learners in this committee is not what it's supposed to be.");

    for (size_t i = 0; i < batchSize; i++)
    {
        if (m_type == kCommitteeAverage)
        {
            (*outputStart) = outputs[0][i];
            for (size_t c = 1; c < outputs.size(); c++)
                s_accum((*outputStart), outputs[c][i]);
            for (size_t j = 0; j < (*outputStart).size(); j++)
                (*outputStart)[j] /= ((f64)m_committee.size());
        }
        else if (m_type == kCommitteeMostConfident)
        {
            size_t mostConfidentIndex = 0;
            for (size_t c = 1; c < outputs.size(); c++)
                if (s_max(outputs[c][i]) > s_max(outputs[mostConfidentIndex][i]))
                    mostConfidentIndex = c;
            (*outputStart) = outputs[mostConfidentIndex][i];
        }
        else
        {
            throw eNotImplemented("Is there a new enum value I haven't handled here yet?");
        }
        outputStart++;
    }
}

void tLearnerCommittee::printLearnerInfo(std::ostream& out) const
{
    for (size_t i = 0; i < m_committee.size(); i++)
    {
        out << "Committee member " << i+1 << ":" << std::endl;
        m_committee[i]->printLearnerInfo(out);
        out << std::endl;
    }
}

std::string tLearnerCommittee::learnerInfoString() const
{
    std::string str = "Committee____";
    str += m_committee[0]->learnerInfoString();
    for (size_t i = 1; i < m_committee.size(); i++)
    {
        str += "__+__";
        str += m_committee[i]->learnerInfoString();
    }
    return str;
}

f64 tLearnerCommittee::calculateError(const tIO& output, const tIO& target)
{
    f64 sum = 0.0;
    for (size_t i = 0; i < m_committee.size(); i++)
        sum += m_committee[i]->calculateError(output, target);
    return sum / ((f64)m_committee.size());
}

f64 tLearnerCommittee::calculateError(const std::vector<tIO>& outputs,
                                      const std::vector<tIO>& targets)
{
    f64 sum = 0.0;
    for (size_t i = 0; i < m_committee.size(); i++)
        sum += m_committee[i]->calculateError(outputs, targets);
    return sum / ((f64)m_committee.size());
}

void tLearnerCommittee::addExample(const tIO& input, const tIO& target)
{
    throw eLogicError("Do not call this method. You cannot train a committee of learners.");
}

void tLearnerCommittee::update()
{
    throw eLogicError("Do not call this method. You cannot train a committee of learners.");
}

void tLearnerCommittee::reset()
{
    throw eLogicError("Do not call this method. You cannot train a committee of learners.");
}


}   // namespace ml
