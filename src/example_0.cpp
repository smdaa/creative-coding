#include "cinder/app/App.h"
#include "cinder/app/RendererGl.h"

using namespace ci;
using namespace ci::app;

class CreativeCodingApp : public App
{
public:
    void setup() override;
    void draw() override;
};

void CreativeCodingApp::setup()
{
    // Your setup code here
}

void CreativeCodingApp::draw()
{
    // Your drawing code here
}

CINDER_APP(CreativeCodingApp, RendererGl)
