#define BOOST_TEST_MODULE test
#include <boost/test/included/unit_test.hpp>
#include "../rnn.cpp"

BOOST_AUTO_TEST_CASE( test )
{
    auto * rnn = new RnnLayer();
    auto * h = new float[3];
    h[0]=3,h[1]=4,h[2]=1;
    BOOST_CHECK_CLOSE(rnn->activateSoftmax(h,3)[0],0.25949646034242,1e-5);
    BOOST_CHECK_CLOSE(rnn->activateSoftmax(h,3)[1],0.70538451269824,1e-5);
    BOOST_CHECK_CLOSE(rnn->activateSoftmax(h,3)[2],0.03511902695934,1e-5);
}