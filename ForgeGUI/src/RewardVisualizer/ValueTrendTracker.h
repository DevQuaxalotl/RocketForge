#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <string>
#include <unordered_map>

namespace ForgeGUI {

/** Short-window change of one value series. */
struct TrendReading {
    /** Smoothed current value minus the smoothed value TREND_WINDOW_SECONDS of game time earlier. */
    float delta = 0.f;

    /** +1 rising, -1 falling, 0 inside the dead-band or without enough history yet. */
    int direction = 0;
};

/**
 * Tracks named value series over game time and reports each one's recent change.
 * Header-only and Qt-free so ForgeTrainingTests can unit-test it without linking ForgeGUI.
 */
class ValueTrendTracker {
public:
    static constexpr double TREND_WINDOW_SECONDS = 0.5;

    /** Each end of the window is averaged over this long, so per-step jitter doesn't read as a trend. */
    static constexpr double SMOOTHING_SECONDS = 0.1;

    /** A change smaller than this fraction of the series' largest magnitude reads as flat. */
    static constexpr float DEAD_BAND_FRACTION = 0.01f;

    /** Clears every series when the episode changes; values across an episode reset aren't comparable. */
    void ObserveEpisode(uint64_t episodeIndex) {
        if (_hasEpisode && episodeIndex == _episodeIndex)
            return;

        _hasEpisode = true;
        _episodeIndex = episodeIndex;
        _series.clear();
    }

    /** A sample not newer than the series' latest is ignored, so a frame delivered twice doesn't distort the trend. */
    void Record(const std::string& key, double gameTime, float value) {
        Series& series = _series[key];
        if (!series.samples.empty() && gameTime <= series.samples.back().time)
            return;

        series.samples.push_back({ gameTime, value });
        series.maxMagnitude = std::max(series.maxMagnitude, std::abs(value));

        const double oldestNeeded = gameTime - TREND_WINDOW_SECONDS - SMOOTHING_SECONDS;
        while (series.samples.size() > 1 && series.samples[1].time <= oldestNeeded)
            series.samples.pop_front();
    }

    TrendReading Get(const std::string& key) const {
        const auto found = _series.find(key);
        if (found == _series.end() || found->second.samples.empty())
            return {};

        const Series& series = found->second;
        const double now = series.samples.back().time;
        const double pastEnd = now - TREND_WINDOW_SECONDS;
        if (series.samples.front().time > pastEnd)
            return {};

        TrendReading reading;
        reading.delta = MeanBetween(series, now - SMOOTHING_SECONDS, now) - MeanBetween(series, pastEnd - SMOOTHING_SECONDS, pastEnd);

        const float deadBand = std::max(DEAD_BAND_FRACTION * series.maxMagnitude, MIN_DEAD_BAND);
        if (std::abs(reading.delta) > deadBand)
            reading.direction = reading.delta > 0.f ? 1 : -1;

        return reading;
    }

    void Clear() {
        _series.clear();
        _hasEpisode = false;
    }

private:
    static constexpr float MIN_DEAD_BAND = 1e-6f;

    struct Sample {
        double time;
        float value;
    };

    struct Series {
        std::deque<Sample> samples;
        float maxMagnitude = 0.f;
    };

    /** Mean of samples in [from, to]; the latest sample at or before `to` when the interval holds none. */
    static float MeanBetween(const Series& series, double from, double to) {
        double sum = 0.0;
        int count = 0;
        float latestBefore = series.samples.front().value;

        for (const Sample& sample : series.samples) {
            if (sample.time > to)
                break;

            latestBefore = sample.value;
            if (sample.time >= from) {
                sum += sample.value;
                count++;
            }
        }

        return count > 0 ? (float)(sum / count) : latestBefore;
    }

    std::unordered_map<std::string, Series> _series;
    uint64_t _episodeIndex = 0;
    bool _hasEpisode = false;
};

} // namespace ForgeGUI
