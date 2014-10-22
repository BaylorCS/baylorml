#include <ml/tTSNE.h>

#include <rho/img/tCanvas.h>

#include "bh_tsne/tsne.h"


namespace ml
{


void tsne(const std::vector<tIO>& originalData, f64 theta, f64 perplexity,
                std::vector<tIO>& reducedDimData)
{
    // Verify that the input is reasonable.
    if (originalData.size() == 0)
        throw eInvalidArgument("There must be at least one original data point.");
    for (size_t i = 1; i < originalData.size(); i++)
        if (originalData[i].size() != originalData[0].size())
            throw eInvalidArgument("All data points must have the same dimensionality.");
    if (originalData[0].size() == 0)
        throw eInvalidArgument("The dimensionality of the data points must be non-zero.");
    if (theta < 0.0 || theta > 1.0)
        throw eInvalidArgument("Theta must be in the range [0.0, 1.0].");

    // Setup the state that the t-SNE algo needs.
    int N = (int) originalData.size();
    int D = (int) originalData[0].size();
    int redDims = 2;
    double* data = (double*) malloc(N * D * sizeof(double));
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < D; j++)
            data[i*D + j] = originalData[i][j];
    }
    double* Y = (double*) malloc(N * redDims * sizeof(double));

    // Run t-SNE.
    TSNE tsneobj;
    tsneobj.run(data, N, D, Y, redDims, perplexity, theta);
    free(data); data = NULL;

    // Copy the results out.
    reducedDimData.resize(N);
    for (int i = 0; i < N; i++)
    {
        reducedDimData[i].resize(redDims);
        for (int j = 0; j < redDims; j++)
            reducedDimData[i][j] = Y[i*redDims + j];
    }
    free(Y); Y = NULL;
}


void plotImages(const std::vector<tIO>& images, bool color, u32 width,
                bool absolute, const std::vector<tIO>& locations,
                u32 destWidth, img::tImage* dest)
{
    // Verify the input.
    if (images.size() != locations.size())
        throw eInvalidArgument("There must be the same number of images as locations.");
    if (images.size() == 0)
        throw eInvalidArgument("There must be at least one image to plot.");
    for (size_t i = 1; i < images.size(); i++)
        if (images[i].size() != images[0].size())
            throw eInvalidArgument("Each image must have the same dimensionality.");
    if (images[0].size() == 0)
        throw eInvalidArgument("The image dimensionality must be non-zero.");
    for (size_t i = 0; i < locations.size(); i++)
        if (locations[i].size() != 2)
            throw eInvalidArgument("Each locations must specify a 2D point.");

    // Normalize the locations to the coordinate space we want.
    f64 xmin = locations[0][0];
    f64 xmax = locations[0][0];
    f64 ymin = locations[0][1];
    f64 ymax = locations[0][1];
    for (size_t i = 1; i < locations.size(); i++)
    {
        xmin = std::min(xmin, locations[i][0]);
        xmax = std::max(xmax, locations[i][0]);
        ymin = std::min(ymin, locations[i][1]);
        ymax = std::max(ymax, locations[i][1]);
    }
    f64 scale = ((f64)destWidth) / (xmax - xmin);
    std::vector<tIO> locs = locations;
    for (size_t i = 0; i < locs.size(); i++)
    {
        locs[i][0] = (locs[i][0] - xmin) * scale;
        locs[i][1] = (locs[i][1] - ymin) * scale;
    }

    // Plot each image onto a canvas.
    u8 bgColor[3] = { 255, 255, 255 };    // white
    img::tCanvas canvas(img::kRGB24, bgColor, 3);
    img::tImage image;
    for (size_t i = 0; i < locs.size(); i++)
    {
        ml::un_examplify(images[i], color, width, absolute, &image);
        canvas.drawImage(&image, (i32)round(locs[i][0]), (i32)round(locs[i][1]));
    }
    canvas.genImage(dest);
}


}  // namespace ml
