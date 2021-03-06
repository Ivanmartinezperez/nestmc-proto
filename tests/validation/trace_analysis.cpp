#include <algorithm>
#include <fstream>
#include <string>

#include <json/json.hpp>

#include "gtest.h"

#include <math.hpp>
#include <simple_sampler.hpp>
#include <util/optional.hpp>
#include <util/partition.hpp>
#include <util/rangeutil.hpp>

#include "trace_analysis.hpp"

namespace nest {
namespace mc {

struct trace_interpolant {
    trace_interpolant(const trace_data& trace): trace_(trace) {}

    double operator()(float t) const {
        if (trace_.empty()) return std::nan("");

        auto tx = times(trace_);
        auto vx = values(trace_);

        // special case for end points
        if (t<tx.front()) return vx.front();
        if (t>=tx.back()) return vx.back();

        auto part = util::partition_view(tx);
        auto i = part.index(t);
        EXPECTS(i != part.npos);
        auto p = part[i];
        return math::lerp(vx[i], vx[i+1], (t-p.first)/(p.second-p.first));
    }

    const trace_data& trace_;
};

double linf_distance(const trace_data& u, const trace_data& ref) {
    trace_interpolant f{ref};

    return util::max_value(
            util::transform_view(u,
                [&](trace_entry x) { return std::abs(x.v-f(x.t)); }));
}

std::vector<trace_peak> local_maxima(const trace_data& u) {
    std::vector<trace_peak> peaks;
    if (u.size()<2) return peaks;

    auto tx = times(u);
    auto vx = values(u);

    int s_prev = math::signum(vx[1]-vx[0]);
    std::size_t i_start = 0;

    for (std::size_t i = 2; i<u.size()-1; ++i) {
        int s = math::signum(vx[i]-vx[i-1]);
        if (s_prev==1 && s==-1) {
            // found peak between i_start and i,
            // observerd peak value at i-1.
            float t0 = tx[i_start];
            float t1 = tx[i];

            peaks.push_back({(t0+t1)/2, vx[i-1], (t1-t0)/2});
        }

        if (s!=0) {
            s_prev = s;
            if (s_prev>0) {
                i_start = i-1;
            }
        }
    }
    return peaks;
}

util::optional<trace_peak> peak_delta(const trace_data& a, const trace_data& b) {
    auto p = local_maxima(a);
    auto q = local_maxima(b);

    if (p.size()!=q.size()) return util::nothing;

    auto max_delta = p[0]-q[0];

    for (std::size_t i = 1u; i<p.size(); ++i) {
        auto delta = p[i]-q[i];
        // pick maximum delta by greatest lower bound on dt
        if (std::abs(delta.t)-delta.t_err>std::abs(max_delta.t)-max_delta.t_err) {
            max_delta = delta;
        }
    }
    return max_delta;
}

} // namespace mc
} // namespace nest

