#pragma once

#include <type_traits>
#include <vector/include/Vector.hpp>

#include "util.hpp"
#include "util/debug.hpp"

namespace nest {
namespace mc {

/// matrix storage structure
template<typename T, typename I>
class matrix {
    public:

    // define basic types
    using value_type = T;
    using size_type  = I;

    // define storage types
    using vector_type            = memory::HostVector<value_type>;
    using vector_view_type       = typename vector_type::view_type;
    using const_vector_view_type = typename vector_type::const_view_type;
    using index_type             = memory::HostVector<size_type>;
    using index_view_type        = typename index_type::view_type;
    using const_index_view_type  = typename index_type::const_view_type;

    matrix() = default;

    /// construct matrix for one or more cells, with combined parent index
    /// and a cell index
    template <
        typename LHS, typename RHS,
        typename = typename
            std::enable_if<
                util::is_container<LHS>::value &&
                util::is_container<RHS>::value
            >
    >
    matrix(LHS&& pi, RHS&& ci)
    :   parent_index_(std::forward<LHS>(pi))
    ,   cell_index_(std::forward<RHS>(ci))
    {
        setup();
    }

    /// construct matrix for a single cell described by a parent index
    template <
        typename IDX,
        typename = typename
            std::enable_if< util::is_container<IDX>::value >
    >
    matrix(IDX&& pi)
    :   parent_index_(std::forward<IDX>(pi))
    ,   cell_index_(2)
    {
        cell_index_[0] = 0;
        cell_index_[1] = size();
        setup();
    }

    /// the dimension of the matrix (i.e. the number of rows or colums)
    std::size_t size() const
    {
        return parent_index_.size();
    }

    /// the total memory used to store the matrix
    std::size_t memory() const
    {
        auto s = 6 * (sizeof(value_type) * size() + sizeof(vector_type));
        s     += sizeof(size_type) * (parent_index_.size() + cell_index_.size())
                + 2*sizeof(index_type);
        s     += sizeof(matrix);
        return s;
    }

    /// the number of cell matrices that have been packed together
    size_type num_cells() const
    {
        return cell_index_.size() - 1;
    }

    /// FIXME : make modparser use the correct accessors (l,d,u,rhs) instead of these
    vector_view_type vec_rhs() { return rhs(); }
    vector_view_type vec_d()   { return d(); }
    vector_view_type vec_v()   { return v(); }

    const_vector_view_type vec_rhs() const { return rhs(); }
    const_vector_view_type vec_d()   const { return d(); }
    const_vector_view_type vec_v()   const { return v(); }

    /// the vector holding the lower part of the matrix
    vector_view_type l() {
        return l_;
    }

    const_vector_view_type l() const {
        return l_;
    }

    /// the vector holding the diagonal of the matrix
    vector_view_type d() {
        return d_;
    }

    const vector_view_type d() const {
        return d_;
    }

    /// the vector holding the upper part of the matrix
    vector_view_type u() {
        return u_;
    }
    const vector_view_type u() const {
        return u_;
    }

    /// the vector holding the right hand side of the linear equation system
    vector_view_type rhs() {
        return rhs_;
    }

    const_vector_view_type rhs() const {
        return rhs_;
    }

    /// the vector holding the solution (voltage)
    vector_view_type v() {
        EXPECTS(has_voltage_);
        return v_;
    }

    const_vector_view_type v() const {
        EXPECTS(has_voltage_);
        return v_;
    }

    /// the vector holding the parent index
    index_view_type p() {
        return parent_index_;
    }

    const_index_view_type p() const {
        return parent_index_;
    }

    /// solve the linear system
    /// upon completion the solution is stored in the RHS storage
    /// and can be accessed via rhs()
    void solve()
    {
        /*
        std::cout << "solving matrix :\n";
        std::cout << "  l   " << l_   << "\n";
        std::cout << "  d   " << d_   << "\n";
        std::cout << "  u   " << u_   << "\n";
        std::cout << "  rhs " << rhs_ << "\n";
        */

        // FIXME make a const view
        index_view_type const& p = parent_index_;
        auto const ncells = num_cells();

        // loop over submatrices
        for (size_type m=0; m<ncells; ++m) {
            auto first = cell_index_[m];
            auto last = cell_index_[m+1];

            // backward sweep
            for(auto i=last-1; i>first; --i) {
                auto factor = l_[i] / d_[i];
                d_[p[i]]   -= factor * u_[i];
                rhs_[p[i]] -= factor * rhs_[i];
            }
            rhs_[first] /= d_[first];

            // forward sweep
            for(auto i=first+1; i<last; ++i) {
                rhs_[i] -= u_[i] * rhs_[p[i]];
                rhs_[i] /= d_[i];
            }
        }
        //std::cout << "  v   " << rhs_ << "\n";
    }

    void add_voltage(vector_view_type v_ext)
    {
        EXPECTS(v_ext.size()==size());

        std::cout << "============ adding voltage" << std::endl;

        v_ = v_ext;
        has_voltage_ = true;
    }

    private:

    /// allocate memory for storing matrix and right hand side vector
    void setup()
    {
        const auto n = size();
        constexpr auto default_value
            = std::numeric_limits<value_type>::quiet_NaN();

        l_   = vector_type(n, default_value);
        d_   = vector_type(n, default_value);
        u_   = vector_type(n, default_value);
        rhs_ = vector_type(n, default_value);
    }

    /// the parent indice that describe matrix structure
    index_type parent_index_;

    /// indexes that point to the start of each cell in the index
    index_type cell_index_;

    /// storage for lower, diagonal, upper and rhs
    vector_type l_;
    vector_type d_;
    vector_type u_;

    /// after calling solve, the solution is stored in rhs_
    vector_type rhs_;
    vector_view_type v_;

    bool has_voltage_=false;
};

} // namespace nest
} // namespace mc
