#pragma once

namespace lattice
{
/// @brief Class for cubic d-dimensional lattices
    class hypercubic : public lattice_base{
    private:
        int _dim = 3;
        int _L = 10;
        
    public:
        ~hypercubic() { DESTRUCTOR_CALL; }
        hypercubic() = default;

        /// @brief Constructor of cubic lattice
        /// @param L linear dimension (lenght of cubic)
        /// @param dim dimensionality (by default 3D)
        hypercubic(int L, bool _bound_cond, int dim = 3)
        {
            CONSTRUCTOR_CALL;
            this->_L = L;
            this->_dim = dim;
            this->volume = (u64)std::pow(this->_L, this->_dim);
            this->boundary_condition = !_bound_cond; // Change boolean, in main part PBC=0, but here PBC=1
            // _debug_start 
            std::cout << "Created lattice of linear size L = " << this->_L << " in dimensions d = " << this->_dim << " with BC = " << this->boundary_condition << std::endl;
        }
        //TODO: constructor to take vector of Lx, Ly, Lz, ... and dim = vector.size(), so both in one code
        //TODO: below all this->_L substitude with this->_dimensions[d] or similar

        /// @brief Get index of position given by input coordinates
        /// @param coordinates vector of coordinates (dimensionality must be same as lattice)
        /// @return index for given coordinates
        virtual u64 get_index(const arma::uvec& coordinates) 
            const 
        {
            _assert_(this->_dim == coordinates.size(), 
                "Size mismatch! Given number of coordinates does not match dimensionality of lattice");
            u64 idx = coordinates(0);
            u64 quotient = this->_L;
            if(this->_dim > 1){
                for(int j = 1; j < this->_dim; j++){
                    idx += coordinates(j) * quotient; //(u64)std::pow(double(this->_L), j - 1.0);
                    quotient *= this->_L;
                }
            }
            return idx;
        }
        
        /// @brief Get coordinates for given index in lattice
        /// @param idx index of position in lattice
        /// @return coordinates at given index
        virtual arma::uvec get_coordinates(u64 idx)
            const 
        {
            arma::uvec coords(this->_dim);
            if(this->_dim == 1){
                coords(0) = idx;
            }
            else{
                for(int d = 0; d < this->_dim; d++){
                    coords(d) = idx % this->_L;
                    idx = idx / this->_L;
                }
            }
            return coords;
        }

        /// @brief Find coordinates of nearest neighbour to input site coords
        /// @param coordinates site coordinates
        /// @return neighbour to input site
        virtual arma::uvec get_nearest_neighbour(const arma::uvec& coordinates)
            const
        {
            arma::uvec coords_neigh = coordinates;
            
            coords_neigh(0) += 1;
            if(this->boundary_condition && coords_neigh(0) >= this->_L)   
                coords_neigh(0) = coords_neigh(0) % this->_L;
        
            return coords_neigh;
        }

        /// @brief Find coordinates of next-nearest neighbour to input site coords
        /// @param coordinates site coordinates
        /// @return next-neighbour to input site
        virtual arma::uvec get_next_nearest_neighbour(const arma::uvec& coordinates)
            const
        {
            arma::uvec coords_neigh = coordinates;
            if(this->_dim == 1){
                coords_neigh(0) += 2;
                if(this->boundary_condition && coords_neigh(0) >= this->_L)   
                    coords_neigh(0) = coords_neigh(0) % this->_L;
            } else {
                for(int d = 0; d <= 1; d++){
                    coords_neigh(d) += 1;
                    if(this->boundary_condition && coords_neigh(d) >= this->_L)   
                        coords_neigh(d) = coords_neigh(d) % this->_L;
                }
            }
            return coords_neigh;
        }
        
        /// @brief Find all nearest neighbours to a given set of coordinates
        /// @param coordinates 
        /// @return vector of indices of neigbours
        virtual std::vector<site_type> get_neighbours(const arma::uvec& coordinates)
            const
        {
            std::vector<site_type> neis;
            std::cout << coordinates.t();
            for(int d = 0; d < this->_dim; d++)
            {
                arma::uvec coords_neigh = coordinates;
                if(coords_neigh(d) == this->_L - 1){
                    if(this->boundary_condition){
                        coords_neigh(d) = (coords_neigh(d) + 1) % this->_L;
                        neis.push_back( get_index(coords_neigh) );
                    }
                } else {
                    coords_neigh(d) = coords_neigh(d) + 1;
                    neis.push_back( get_index(coords_neigh) );
                }
                _extra_debug( std::cout << "Neighbour: " << coords_neigh.t(); )

                coords_neigh = coordinates;
                if(coords_neigh(d) == 0){
                    if(this->boundary_condition){
                        coords_neigh(d) = coords_neigh(d) + this->_L - 1;
                        neis.push_back( get_index(coords_neigh) );
                    }
                } else {
                    coords_neigh(d) = coords_neigh(d) - 1;
                    neis.push_back( get_index(coords_neigh) );
                }
                _extra_debug( std::cout << "Neighbour: " <<  coords_neigh.t(); )
            }
            return neis;
        }

        /// @brief Find all nearest neighbours of a given site
        /// @param site input site to find neighbours
        /// @return vector of indices of neigbours
        virtual std::vector<site_type> get_neighbours(site_type site)
            const override
        {
            auto coordinates = get_coordinates(site);
            return get_neighbours(coordinates);
        }

        /// @brief Find a particular nearest neighbour (in x-direction)
        /// @param site input site to find neighbours
        /// @return nearest neighbour in x
        virtual site_type get_nearest_neighbour(site_type site)
            const override
        {
            auto coordinates = get_coordinates(site);
            return get_index( get_nearest_neighbour(coordinates) );
        }
        
        /// @brief Find a particular next nearest neighbour (in x-direction)
        /// @param site input site to find next nearest neighbours
        /// @return next nearest neighbour in x
        virtual site_type get_next_nearest_neighbour(site_type site)
            const override
        {
            auto coordinates = get_coordinates(site);
            return get_index( get_next_nearest_neighbour(coordinates) );
        }

    
    };


    // /// @brief Class for cubic d-dimensional lattices
    // class general_lattice : public lattice_base{
    // private:
    //     arma::uvec dimensions;
    //     int _dim = 3;
    // public:
    //     ~general_lattice() { DESTRUCTOR_CALL; }
    //     /// @brief Constructor of cubic lattice
    //     /// @param dimens Vector of dimensions
    //     general_lattice(const arma::uvec& dimens){
    //         CONSTRUCTOR_CALL;
    //         this->dimensions = dimens;
    //         this->volume = arma::prod(this->dimensions);
    //         this->_dim = this->dimensions.size();
    //     }
    //     /// @brief Get index of position given by input coordinates
    //     /// @param coordinates vector of coordinates (dimensionality must be same as lattice)
    //     /// @return index for given coordinates
    //     u64 get_index(const arma::uvec& coordinates){
    //         _assert_(this->_dim == coordinates.size(), "Size mismatch! Given number of coordinates does not match dimensionality of lattice");
    //         u64 idx = coordinates(0);
    //         for(int j = 1; j < this->_dim; j++)
    //             idx += coordinates(j) * arma::prod(this->dimensions.rows(0, j - 1));
    //         return idx;
    //     }   
    //     /// @brief Get coordinates for given index in lattice
    //     /// @param idx index of position in lattice
    //     /// @return coordinates at given index
    //     arma::uvec get_coordinates(u64 idx){
    //         arma::uvec coords(this->_dim);       
    //     }
    // };
}