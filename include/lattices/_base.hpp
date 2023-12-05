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

        virtual arma::uvec get_nearest_neighbour(const arma::uvec& coordinates)      const { return arma::uvec(); };
        virtual arma::uvec get_next_nearest_neighbour(const arma::uvec& coordinates) const { return arma::uvec(); };
 
        virtual u64 get_index(const arma::uvec& coordinates)                         const { return 0; };
        virtual arma::uvec get_coordinates(u64 idx)                                  const { return arma::uvec(); };
        
    };
    inline lattice_base::~lattice_base(){}


    /// @brief Class for cubic d-dimensional lattices
    class hypercubic : public lattice_base{
    private:
        int _dim = 3;
        int _L;
        u64 volume;
        
    public:
        ~hypercubic() { DESTRUCTOR_CALL; }

        /// @brief Constructor of cubic lattice
        /// @param L linear dimension (lenght of cubic)
        /// @param dim dimensionality (by default 3D)
        hypercubic(int L, int dim = 3)
        {
            CONSTRUCTOR_CALL;
            this->_L = L;
            this->_dim = dim;
            this->volume = (u64)std::pow(this->_L, this->_dim);
        }

        /// @brief Get index of position given by input coordinates
        /// @param coordinates vector of coordinates (dimensionality must be same as lattice)
        /// @return index for given coordinates
        virtual u64 get_index(const arma::uvec& coordinates) 
            const override 
        {
            _assert_(this->_dim == coordinates.size(), 
                "Size mismatch! Given number of coordinates does not match dimensionality of lattice");
            u64 idx = coordinates(0);
            u64 quotient = 1;
            for(int j = 1; j < this->_dim; j++){
                idx += coordinates(j) * quotient; //(u64)std::pow(double(this->_L), j - 1.0);
                quotient *= this->_L;
            }
            return idx;
        }
        
        /// @brief Get coordinates for given index in lattice
        /// @param idx index of position in lattice
        /// @return coordinates at given index
        virtual arma::uvec get_coordinates(u64 idx)
            const override 
        {
            arma::uvec coords(this->_dim);
            int ii = 0;
            while(idx != 1){
                coords(ii++) = idx % this->_L;
                idx = idx / this->_L;
            }
            return coords;
        }

        virtual std::vector<site_type> get_neighbours(site_type site)
            const override
        {
            return std::vector<site_type>();
        }



        virtual site_type get_nearest_neighbour(site_type site)
            const override
        {
            if(site % this->_L == this->_L - 1){
                if(this->boundary_condition) return site - this->_L + 1;
                else                         return -1;
            } else 
                return site + 1;
        }
        
        /// @brief 
        /// @param site 
        /// @return 
        virtual site_type get_next_nearest_neighbour(site_type site)
            const override
        {
            if(this->_dim == 1){
                if(site % this->_L >= this->_L - 2){
                    if(this->boundary_condition) return site - this->_L + 2;
                    else                         return -1;
                } else 
                    return site + 2;
            } else {
                return -1;
            }
        }

        /// @brief Find coordinates of nearest neighbour to input site coords
        /// @param coordinates site coordinates
        /// @return neighbour to input site
        virtual arma::uvec get_nearest_neighbour(const arma::uvec& coordinates)
            const override
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
            const override
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
#include "lattice1d.hpp"
#include "lattice2d.hpp"
#include "lattice3d.hpp"
