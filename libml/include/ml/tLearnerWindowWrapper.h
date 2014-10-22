#ifndef __ml_tLearnerWindowWrapper_h__
#define __ml_tLearnerWindowWrapper_h__


#include <ml/rhocompat.h>
#include <ml/common.h>

#include <rho/app/tMainLoopGLFW.h>
#include <rho/app/tSimpleImageWindow.h>
#include <rho/algo/string_util.h>

#include <iostream>
#include <sstream>
#include <string>


namespace ml
{


class tLearnerWindowWrapper : public iEZTrainObserver
{
    public:

        tLearnerWindowWrapper(std::string initTitle, bool isInputImageColor, u32 inputImageWidth, u32 isAbsolute)
            : m_mainLoop(),
              m_window(NULL),
              m_iscolor(isInputImageColor),
              m_width(inputImageWidth),
              m_absolute(isAbsolute)
        {
            m_window = new app::tSimpleImageWindow(300, 100, initTitle);
            m_mainLoop.addWindow(m_window);
            m_mainLoop.once();
        }

        ~tLearnerWindowWrapper()
        {
            m_window->setWindowShouldClose();
            m_mainLoop.loop();
            m_window = NULL;
        }

        /////////////////////////////////////////////////////////////////////
        // iTrainObserver interface:
        /////////////////////////////////////////////////////////////////////

        bool didUpdate(iLearner* learner, const std::vector<tIO>& mostRecentBatch)
        {
            m_mainLoop.once();
            return true;
        }

        /////////////////////////////////////////////////////////////////////
        // iEZTrainObserver interface:
        /////////////////////////////////////////////////////////////////////

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
                            f64 epochTrainTimeInSeconds)
        {
            // Get a reference to the simple window.
            refc<app::tWindowGLFW> windowRef = m_window;
            app::tWindowGLFW* windowPtr = windowRef;
            app::tSimpleImageWindow& window = *(dynamic_cast<app::tSimpleImageWindow*>(windowPtr));

            // If the window is closed, don't do anything expensive.
            if (window.shouldWindowClose())
            {
                window.setImage(NULL);
                return true;
            }

            // Set the window title.
            std::ostringstream out;
            out << algo::replace(learner->learnerInfoString(), "__", "    ") << "    ";
            out << "fold: " << foldIndex+1 << "    ";
            out << "epoch: " << epochsCompleted << "    ";
            out << "(last epoch took " << epochTrainTimeInSeconds << " seconds)";
            window.setTitle(out.str());

            // Set the window contents.
            img::tImage image;
            visualize(learner, trainInputs.front(), m_iscolor, m_width, m_absolute, &image);
            window.setImage(&image);

            return true;
        }

        void didFinishTraining(iLearner* learner,
                               u32 epochsCompleted,
                               u32 foldIndex, u32 numFolds,
                               const std::vector< tIO >& trainInputs,
                               const std::vector< tIO >& trainTargets,
                               const std::vector< tIO >& testInputs,
                               const std::vector< tIO >& testTargets,
                               f64 trainingTimeInSeconds)
        {
            // Nothing...
        }

    private:

        app::tMainLoopGLFW m_mainLoop;
        refc<app::tWindowGLFW> m_window;

        bool m_iscolor;
        u32  m_width;
        bool m_absolute;
};


}   // namespace ml


#endif   // __ml_tLearnerWindowWrapper_h__
