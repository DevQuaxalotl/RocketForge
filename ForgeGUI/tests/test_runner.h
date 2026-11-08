#pragma once

#include <functional>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace RocketForgeTests {

	struct TestCase {
		std::string name;
		std::function<void()> fn;
	};

	std::vector<TestCase>& Registry();

	struct AutoRegister {
		AutoRegister(const std::string& name, std::function<void()> fn) {
			Registry().push_back({ name, std::move(fn) });
		}
	};

	struct AssertionFailure {
		std::string message;
	};

	inline void FailAssert(const std::string& message) {
		throw AssertionFailure{ message };
	}

	template <typename T, typename = void>
	struct Printable {
		static std::string Str(const T&) { return "<unprintable>"; }
	};

	template <typename T>
	struct Printable<T, std::void_t<decltype(std::declval<std::ostream&>() << std::declval<const T&>())>> {
		static std::string Str(const T& value) {
			std::ostringstream oss;
			oss << value;
			return oss.str();
		}
	};

	template <typename A, typename B>
	void AssertEqImpl(const A& actual, const B& expected, const char* actualExpr, const char* expectedExpr, const char* file, int line) {
		if (!(actual == expected)) {
			std::ostringstream oss;
			oss << file << ":" << line << ": ASSERT_EQ failed:\n"
				<< "  " << actualExpr << " => " << Printable<A>::Str(actual) << "\n"
				<< "  " << expectedExpr << " => " << Printable<B>::Str(expected);
			FailAssert(oss.str());
		}
	}

	inline void AssertTrueImpl(bool actual, const char* expr, const char* file, int line) {
		if (!actual) {
			std::ostringstream oss;
			oss << file << ":" << line << ": ASSERT_TRUE(" << expr << ") failed";
			FailAssert(oss.str());
		}
	}

	inline void AssertFalseImpl(bool actual, const char* expr, const char* file, int line) {
		if (actual) {
			std::ostringstream oss;
			oss << file << ":" << line << ": ASSERT_FALSE(" << expr << ") failed";
			FailAssert(oss.str());
		}
	}
}

#define ROCKETFORGE_TEST(name) \
	static void name(); \
	static RocketForgeTests::AutoRegister _reg_##name(#name, name); \
	static void name()

#define ASSERT_EQ(a, b) RocketForgeTests::AssertEqImpl((a), (b), #a, #b, __FILE__, __LINE__)
#define ASSERT_TRUE(x)  RocketForgeTests::AssertTrueImpl((x), #x, __FILE__, __LINE__)
#define ASSERT_FALSE(x) RocketForgeTests::AssertFalseImpl((x), #x, __FILE__, __LINE__)
#define FAIL(msg)       RocketForgeTests::FailAssert((msg))
