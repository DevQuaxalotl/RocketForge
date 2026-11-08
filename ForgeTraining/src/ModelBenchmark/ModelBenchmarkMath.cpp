#include "ModelBenchmarkInternal.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <numeric>
#include <sstream>

namespace RocketForgeTraining::ModelBenchmarkInternal
{
	double InverseNormalCDF(double p) {
			p = std::clamp(p, 1e-9, 1.0 - 1e-9);
			static constexpr double a[] = {
				-3.969683028665376e+01, 2.209460984245205e+02,
				-2.759285104469687e+02, 1.383577518672690e+02,
				-3.066479806614716e+01, 2.506628277459239e+00
			};
			static constexpr double b[] = {
				-5.447609879822406e+01, 1.615858368580409e+02,
				-1.556989798598866e+02, 6.680131188771972e+01,
				-1.328068155288572e+01
			};
			static constexpr double c[] = {
				-7.784894002430293e-03, -3.223964580411365e-01,
				-2.400758277161838e+00, -2.549732539343734e+00,
				4.374664141464968e+00, 2.938163982698783e+00
			};
			static constexpr double d[] = {
				7.784695709041462e-03, 3.224671290700398e-01,
				2.445134137142996e+00, 3.754408661907416e+00
			};
			const double plow = 0.02425;
			const double phigh = 1.0 - plow;
			if (p < plow) {
				double q = std::sqrt(-2.0 * std::log(p));
				return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
					((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
			}
			if (p > phigh) {
				double q = std::sqrt(-2.0 * std::log(1.0 - p));
				return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
					((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
			}
			double q = p - 0.5;
			double r = q * q;
			return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
				(((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0);
	}

	namespace
	{
		std::vector<double> SolveLinearSystem(std::vector<std::vector<double>> a, std::vector<double> b) {
			const int n = (int)b.size();
			for (int col = 0; col < n; col++) {
				int pivot = col;
				for (int row = col + 1; row < n; row++)
					if (std::abs(a[row][col]) > std::abs(a[pivot][col]))
						pivot = row;
				if (std::abs(a[pivot][col]) < 1e-9)
					continue;
				if (pivot != col) {
					std::swap(a[pivot], a[col]);
					std::swap(b[pivot], b[col]);
				}
				double div = a[col][col];
				for (int j = col; j < n; j++)
					a[col][j] /= div;
				b[col] /= div;
				for (int row = 0; row < n; row++) {
					if (row == col)
						continue;
					double factor = a[row][col];
					for (int j = col; j < n; j++)
						a[row][j] -= factor * a[col][j];
					b[row] -= factor * b[col];
				}
			}
			return b;
		}

		std::vector<std::vector<double>> InvertMatrix(std::vector<std::vector<double>> a) {
			const int n = (int)a.size();
			std::vector<std::vector<double>> inv(n, std::vector<double>(n, 0.0));
			for (int i = 0; i < n; i++)
				inv[i][i] = 1.0;
			for (int col = 0; col < n; col++) {
				int pivot = col;
				for (int row = col + 1; row < n; row++)
					if (std::abs(a[row][col]) > std::abs(a[pivot][col]))
						pivot = row;
				if (std::abs(a[pivot][col]) < 1e-9) {
					a[col][col] += 1e-6;
					pivot = col;
				}
				if (pivot != col) {
					std::swap(a[pivot], a[col]);
					std::swap(inv[pivot], inv[col]);
				}
				double div = a[col][col];
				for (int j = 0; j < n; j++) {
					a[col][j] /= div;
					inv[col][j] /= div;
				}
				for (int row = 0; row < n; row++) {
					if (row == col)
						continue;
					double factor = a[row][col];
					for (int j = 0; j < n; j++) {
						a[row][j] -= factor * a[col][j];
						inv[row][j] -= factor * inv[col][j];
					}
				}
			}
			return inv;
		}

		std::string FormatPercent(double value) {
			std::ostringstream out;
			if (value > 0.999)
				out << ">99.9%";
			else
				out << std::fixed << std::setprecision(1) << (value * 100.0) << "%";
			return out.str();
		}
	}

	double ClampProbability(double p) {
		return std::clamp(p, 1e-4, 1.0 - 1e-4);
	}

	double NormalCDF(double x) {
		return 0.5 * std::erfc(-x / std::sqrt(2.0));
	}

	EloFit FitElo(int numModels, const std::vector<BenchmarkOutcome>& outcomes) {
		EloFit result = {};
		result.elo.assign(numModels, 0.0);
		result.covariance.assign(numModels, std::vector<double>(numModels, 0.0));
		if (numModels <= 1)
			return result;

		const int dim = numModels - 1;
		std::vector<double> theta(dim, 0.0);
		std::vector<std::vector<double>> lastInfo(dim, std::vector<double>(dim, 0.0));

		for (int iter = 0; iter < 60; iter++) {
			std::vector<double> grad(dim, 0.0);
			std::vector<std::vector<double>> info(dim, std::vector<double>(dim, 0.0));
			for (const BenchmarkOutcome& outcome : outcomes) {
				if (outcome.winner < 0 || outcome.loser < 0 || outcome.winner == outcome.loser)
					continue;
				double tw = outcome.winner < dim ? theta[outcome.winner] : 0.0;
				double tl = outcome.loser < dim ? theta[outcome.loser] : 0.0;
				double p = ClampProbability(1.0 / (1.0 + std::exp(-(tw - tl))));
				double w = std::max(1e-6, outcome.weight);
				if (outcome.winner < dim)
					grad[outcome.winner] += w * (1.0 - p);
				if (outcome.loser < dim)
					grad[outcome.loser] -= w * (1.0 - p);
				double v = w * p * (1.0 - p);
				if (outcome.winner < dim)
					info[outcome.winner][outcome.winner] += v;
				if (outcome.loser < dim)
					info[outcome.loser][outcome.loser] += v;
				if (outcome.winner < dim && outcome.loser < dim) {
					info[outcome.winner][outcome.loser] -= v;
					info[outcome.loser][outcome.winner] -= v;
				}
			}
			for (int i = 0; i < dim; i++)
				info[i][i] += 1e-4;
			std::vector<double> step = SolveLinearSystem(info, grad);
			double maxStep = 0.0;
			for (int i = 0; i < dim; i++) {
				step[i] = std::clamp(step[i], -1.0, 1.0);
				theta[i] += step[i];
				maxStep = std::max(maxStep, std::abs(step[i]));
			}
			lastInfo = std::move(info);
			if (maxStep < 1e-5)
				break;
		}

		std::vector<double> elo(numModels, 0.0);
		for (int i = 0; i < dim; i++)
			elo[i] = theta[i] * LOGIT_TO_ELO;
		double mean = std::accumulate(elo.begin(), elo.end(), 0.0) / std::max(1, numModels);
		for (double& value : elo)
			value -= mean;
		result.elo = std::move(elo);

		std::vector<std::vector<double>> inv = InvertMatrix(lastInfo);
		std::vector<std::vector<double>> reducedCov(dim, std::vector<double>(dim, 0.0));
		for (int i = 0; i < dim; i++)
			for (int j = 0; j < dim; j++)
				reducedCov[i][j] = inv[i][j] * LOGIT_TO_ELO * LOGIT_TO_ELO;

		// Convert the reduced covariance (with the last model fixed at zero before
		// centering) into the covariance of the displayed mean-centered Elo vector.
		std::vector<double> rowSums(dim, 0.0);
		double totalSum = 0.0;
		for (int i = 0; i < dim; i++) {
			for (int j = 0; j < dim; j++) {
				rowSums[i] += reducedCov[i][j];
				totalSum += reducedCov[i][j];
			}
		}

		const double numModelsD = std::max(1, numModels);
		for (int i = 0; i < numModels; i++) {
			for (int j = 0; j < numModels; j++) {
				double value = totalSum / (numModelsD * numModelsD);
				if (i < dim)
					value -= rowSums[i] / numModelsD;
				if (j < dim)
					value -= rowSums[j] / numModelsD;
				if (i < dim && j < dim)
					value += reducedCov[i][j];
				result.covariance[i][j] = value;
			}
		}
		return result;
	}

	double DifferenceVariance(const std::vector<std::vector<double>>& covariance, int a, int b) {
		if (a < 0 || b < 0 || a >= (int)covariance.size() || b >= (int)covariance.size())
			return 1e9;
		double var = covariance[a][a] + covariance[b][b] - 2.0 * covariance[a][b];
		return std::max(var, 1e-6);
	}

	double ConfidenceBetter(const EloFit& fit, int a, int b) {
		double diff = fit.elo[a] - fit.elo[b];
		double se = std::sqrt(DifferenceVariance(fit.covariance, a, b));
		return NormalCDF(diff / se);
	}

	std::string FormatConfidenceList(
		const EloFit& fit,
		const std::vector<std::string>& names,
		int modelIdx) {
		std::ostringstream out;
		std::vector<int> lowerRatedModels;
		lowerRatedModels.reserve(names.size());
		for (int other = 0; other < (int)names.size(); other++) {
			if (other != modelIdx && fit.elo[modelIdx] > fit.elo[other])
				lowerRatedModels.push_back(other);
		}
		std::sort(lowerRatedModels.begin(), lowerRatedModels.end(), [&](int a, int b) {
			if (fit.elo[a] != fit.elo[b])
				return fit.elo[a] > fit.elo[b];
			return names[a] < names[b];
		});

		bool any = false;
		for (int other : lowerRatedModels) {
			double conf = ConfidenceBetter(fit, modelIdx, other);
			if (any)
				out << ", ";
			out << names[other] << " (";
			if (conf > 0.999)
				out << ">99.9%";
			else
				out << std::fixed << std::setprecision(1) << (conf * 100.0) << "%";
			out << ")";
			any = true;
		}
		if (!any)
			return "--";
		return out.str();
	}

	EloFit MakeRelative(EloFit fit) {
		if (fit.elo.empty())
			return fit;
		double lowest = *std::min_element(fit.elo.begin(), fit.elo.end());
		for (double& elo : fit.elo)
			elo -= lowest;
		return fit;
	}

	void AnchorFitsToOverallFloor(
		EloFit& overallFit,
		EloFit& kickoffFit,
		EloFit& postKickoffFit) {
		if (overallFit.elo.empty())
			return;

		const double overallFloor =
			*std::min_element(overallFit.elo.begin(), overallFit.elo.end());
		for (EloFit* fit : { &overallFit, &kickoffFit, &postKickoffFit }) {
			for (double& elo : fit->elo)
				elo -= overallFloor;
		}
	}

	std::vector<int> BuildRankOrderFromFit(int numModels, const EloFit& fit, const std::vector<std::string>& names) {
		std::vector<int> order(numModels);
		std::iota(order.begin(), order.end(), 0);
		if (numModels <= 1)
			return order;

		std::sort(order.begin(), order.end(), [&](int a, int b) {
			if (std::abs(fit.elo[a] - fit.elo[b]) > 1e-6)
				return fit.elo[a] > fit.elo[b];
			return names[a] < names[b];
		});
		return order;
	}

	std::vector<int> BuildRankOrder(
		int numModels,
		const std::vector<BenchmarkOutcome>& outcomes,
		const std::vector<std::string>& names) {
		if (numModels <= 1 || outcomes.empty()
			|| !HasConnectedOutcomeGraph(numModels, outcomes)) {
			std::vector<int> order(numModels);
			std::iota(order.begin(), order.end(), 0);
			std::sort(order.begin(), order.end(), [&](int a, int b) {
				return names[a] < names[b];
			});
			return order;
		}
		return BuildRankOrderFromFit(numModels, MakeRelative(FitElo(numModels, outcomes)), names);
	}

	LiveEloEstimate BuildLiveEloEstimate(
		int numModels,
		const std::vector<BenchmarkOutcome>& outcomes,
		const std::vector<std::string>& names) {
		LiveEloEstimate estimate;
		estimate.hasEvidence = numModels > 1 && !outcomes.empty();
		estimate.connected = estimate.hasEvidence && HasConnectedOutcomeGraph(numModels, outcomes);
		if (estimate.connected) {
			estimate.fit = MakeRelative(FitElo(numModels, outcomes));
			estimate.order = BuildRankOrderFromFit(numModels, estimate.fit, names);
		} else {
			estimate.order.resize(numModels);
			std::iota(estimate.order.begin(), estimate.order.end(), 0);
			std::sort(estimate.order.begin(), estimate.order.end(), [&](int a, int b) {
				return names[a] < names[b];
			});
		}

		return estimate;
	}

	std::string FormatLiveEstimateSummary(
		const LiveEloEstimate& estimate,
		const std::vector<std::string>& names,
		double eloRangeConfidence) {
		if (!estimate.hasEvidence)
			return "Benchmark estimates:\n  collecting evidence";
		if (!estimate.connected)
			return "Benchmark estimates:\n  collecting evidence to connect every model";

		const EloFit& fit = estimate.fit;
		const std::vector<int>& order = estimate.order;
		const double z = ModelBenchmarkMath::ConfidenceToZ(eloRangeConfidence);

		std::ostringstream out;
		out << "Benchmark Elo:";
		for (int i = 0; i < (int)order.size(); i++) {
			int modelIdx = order[i];
			double se = std::sqrt(std::max(1e-6, fit.covariance[modelIdx][modelIdx]));
			double low = fit.elo[modelIdx] - z * se;
			double high = fit.elo[modelIdx] + z * se;
			out << "\n  " << names[modelIdx]
				<< " +" << std::fixed << std::setprecision(0) << fit.elo[modelIdx]
				<< " [" << low << ", " << high << "]";
		}

		out << "\nBenchmark Ranking Confidence:";
		if (order.size() < 2) {
			out << "\n  collecting evidence";
			return out.str();
		}
		for (int i = 0; i + 1 < (int)order.size(); i++) {
			int better = order[i];
			int worse = order[i + 1];
			out << "\n  " << names[better] << " > " << names[worse] << " " << FormatPercent(ConfidenceBetter(fit, better, worse));
		}

		return out.str();
	}

	std::string FormatLiveEstimateSummary(
		int numModels,
		const std::vector<BenchmarkOutcome>& outcomes,
		const std::vector<std::string>& names,
		double eloRangeConfidence) {
		return FormatLiveEstimateSummary(BuildLiveEloEstimate(numModels, outcomes, names), names, eloRangeConfidence);
	}

	bool AllRangesSmallEnough(const EloFit& fit, double confidence, double maxRange) {
		if (fit.elo.empty())
			return false;
		double z = ModelBenchmarkMath::ConfidenceToZ(confidence);
		for (int i = 0; i < (int)fit.elo.size(); i++) {
			double se = std::sqrt(std::max(1e-6, fit.covariance[i][i]));
			if (2.0 * z * se > maxRange)
				return false;
		}
		return true;
	}

	bool HasMinimumRatedOutcomesForEarlyStop(int ratedOutcomeCount, int matchupCount) {
		return matchupCount > 0 && ratedOutcomeCount >= matchupCount * 2;
	}

	bool HasConnectedOutcomeGraph(int numModels, const std::vector<BenchmarkOutcome>& outcomes) {
		if (numModels < 2)
			return false;

		std::vector<std::vector<int>> neighbors(numModels);
		for (const BenchmarkOutcome& outcome : outcomes) {
			if (outcome.winner < 0 || outcome.winner >= numModels
				|| outcome.loser < 0 || outcome.loser >= numModels
				|| outcome.winner == outcome.loser) {
				continue;
			}
			neighbors[outcome.winner].push_back(outcome.loser);
			neighbors[outcome.loser].push_back(outcome.winner);
		}

		std::vector<bool> visited(numModels, false);
		std::vector<int> pending = { 0 };
		visited[0] = true;
		while (!pending.empty()) {
			const int model = pending.back();
			pending.pop_back();
			for (int neighbor : neighbors[model]) {
				if (visited[neighbor])
					continue;
				visited[neighbor] = true;
				pending.push_back(neighbor);
			}
		}
		return std::all_of(visited.begin(), visited.end(), [](bool value) { return value; });
	}

	int CountModelOutcomes(const std::vector<BenchmarkOutcome>& outcomes, int modelIdx) {
		return (int)std::count_if(
			outcomes.begin(),
			outcomes.end(),
			[modelIdx](const BenchmarkOutcome& outcome) {
				return outcome.winner == modelIdx || outcome.loser == modelIdx;
			});
	}
}

double RocketForgeTraining::ModelBenchmarkMath::ConfidenceToZ(double confidence) {
	confidence = std::clamp(confidence, 0.5, 0.9999);
	return ModelBenchmarkInternal::InverseNormalCDF(0.5 + confidence * 0.5);
}

double RocketForgeTraining::ModelBenchmarkMath::EloFromWinProbability(double winProbability) {
	winProbability = ModelBenchmarkInternal::ClampProbability(winProbability);
	return 400.0 * std::log10(winProbability / (1.0 - winProbability));
}

bool RocketForgeTraining::ModelBenchmarkMath::IsFullRankingKnown(
	const std::vector<double>& elo,
	const std::vector<std::vector<double>>& covariance,
	double confidence) {
	if (elo.size() < 2)
		return false;
	std::vector<int> order(elo.size());
	std::iota(order.begin(), order.end(), 0);
	std::sort(order.begin(), order.end(), [&](int a, int b) {
		return elo[a] > elo[b];
	});
	for (int i = 0; i + 1 < (int)order.size(); i++) {
		int a = order[i];
		int b = order[i + 1];
		double se = std::sqrt(ModelBenchmarkInternal::DifferenceVariance(covariance, a, b));
		if (ModelBenchmarkInternal::NormalCDF((elo[a] - elo[b]) / se) < confidence)
			return false;
	}
	return true;
}
