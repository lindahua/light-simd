/**
 * @file test_main.cpp
 *
 * The main entry of a test program
 *
 * @author Dahua Lin
 */

#include <light_test/std_test_mon.h>
#include "test_aux.h"

#ifdef _MSC_VER
#pragma warning(disable:4100)
#endif

using namespace ltest;

ltest::test_suite lsimd::lsimd_main_suite("Main");

int main(int argc, char *argv[])
{
	lsimd::add_test_packs();

	if (std_test_main(lsimd::lsimd_main_suite))
	{
		return 0;
	}
	else
	{
		return -1;
	}
}




