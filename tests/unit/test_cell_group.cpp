#include "gtest.h"

#include <cell_group.hpp>
#include <common_types.hpp>
#include <fvm_cell.hpp>
#include <util/rangeutil.hpp>

#include "../test_common_cells.hpp"

nest::mc::cell make_cell() {
    using namespace nest::mc;

    nest::mc::cell cell = make_cell_ball_and_stick();

    cell.add_detector({0, 0}, 0);
    cell.segment(1)->set_compartments(101);

    return cell;
}

TEST(cell_group, test)
{
    using namespace nest::mc;

    using cell_group_type = cell_group<fvm::fvm_cell<double, cell_local_size_type>>;
    auto group = cell_group_type{0, util::singleton_view(make_cell())};

    group.advance(50, 0.01);

    // a bit lame...
    EXPECT_EQ(group.spikes().size(), 4u);
}

TEST(cell_group, sources)
{
    using namespace nest::mc;

    // TODO: extend to multi-cell cell groups when the time comes

    using cell_group_type = cell_group<fvm::fvm_cell<double, cell_local_size_type>>;

    auto cell = make_cell();
    EXPECT_EQ(cell.detectors().size(), 1u);
    // add another detector on the cell to make things more interesting
    cell.add_detector({1, 0.3}, 2.3);

    cell_gid_type first_gid = 37u;
    auto group = cell_group_type{first_gid, util::singleton_view(cell)};

    // expect group sources to be lexicographically sorted by source id
    // with gids in cell group's range and indices starting from zero

    const auto& sources = group.spike_sources();
    for (unsigned i = 0; i<sources.size(); ++i) {
        auto id = sources[i].source_id;
        if (i==0) {
            EXPECT_EQ(id.gid, first_gid);
            EXPECT_EQ(id.index, 0u);
        }
        else {
            auto prev = sources[i-1].source_id;
            EXPECT_GT(id, prev);
            EXPECT_EQ(id.index, id.gid==prev.gid? prev.index+1: 0u);
        }
    }
}
