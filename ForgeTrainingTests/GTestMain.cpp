#include "TestSupport.h"

#ifdef ROCKETFORGE_TESTS_HAVE_QT
#include <QCoreApplication>
#endif

int main(int argc, char** argv) {
	::testing::InitGoogleTest(&argc, argv);
	RocketForgeTests::InstallTestEnvironment();

#ifdef ROCKETFORGE_TESTS_HAVE_QT
	// Qt SQL loads its drivers through the application instance.
	// Qt keeps a reference to argc, so it is constructed after GoogleTest has stripped its own flags.
	QCoreApplication app(argc, argv);
#endif

	return RUN_ALL_TESTS();
}
