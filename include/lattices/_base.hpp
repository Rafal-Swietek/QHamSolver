#pragma once

namespace lattice{


    using site_type = signed long long;

    class lattice_base{
    protected:
        bool boundary_condition = true;    //<! boundary condition: true = PBC
    public:

        site_type volume;
        virtual ~lattice_base() = 0;
        virtual std::vector<site_type> get_neighbours(site_type site)                const = 0;

        virtual site_type get_nearest_neighbour(site_type site)                      const = 0;
        virtual site_type get_next_nearest_neighbour(site_type site)                 const = 0;
        
    };
    inline lattice_base::~lattice_base(){}

}
#include "lattice1d.hpp"
#include "lattice2d.hpp"
#include "lattice3d.hpp"
#include "hypercubic.hpp"