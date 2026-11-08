#include <gtest/gtest.h>

#include <GigaLearnCPP/Util/MetricSender.h>

TEST(MetricSender, NextRunTimestepIsLocalOnly)
{
	ASSERT_TRUE(GGL::MetricSender::IsLocalOnlyMetric("SkillTest/NextRunTimestep"));
	ASSERT_FALSE(GGL::MetricSender::IsLocalOnlyMetric("SkillTest/Rating/1v1"));
}
