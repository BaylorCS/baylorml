#include <ml/iLearner.h>

#include <ml/tANN.h>
#include <ml/tCNN.h>


namespace ml
{


iLearner* iLearner::newDeserializedLearner(iReadable* readable)
{
    iLearner* learner = NULL;

    u32 type; rho::unpack(readable, type);

    switch (type)
    {
        case 1:
            learner = new tANN(readable);
            break;

        case 2:
            learner = new tCNN(readable);
            break;

        default:
            throw eRuntimeError("The serialized vector is of an unknown type of learner.");
            break;
    }

    return learner;
}


void iLearner::serializeLearner(iLearner* learner, iWritable* writable)
{
    if (learner == NULL)
        throw eNullPointer("learner must not be NULL");

    tANN* ann = dynamic_cast<tANN*>(learner);
    tCNN* cnn = dynamic_cast<tCNN*>(learner);

    if (ann)
    {
        rho::pack(writable, (u32)1);
        ann->pack(writable);
    }

    else if (cnn)
    {
        rho::pack(writable, (u32)2);
        cnn->pack(writable);
    }

    else
    {
        throw eInvalidArgument("Cannot serialize the given learner because it has an unknown type.");
    }
}


}   // namespace ml
