#pragma once

namespace lattice{

    class lattice1D : public lattice_base{
    private:
        site_type L;
        
    public:

        //---------------------------------------------- CONSTRUCTORS
        lattice1D() = default;

        lattice1D(site_type size, bool _bound_cond = false)
        {
            this->L = size;
            this->boundary_condition = !_bound_cond;
            this->volume = this->L;
        };

        //---------------------------------------------- NEIGHBOURS
        /// @brief Get nearest neighbour to given input site (to the right of input site)
        /// @param site site index
        /// @return nearest-neighbour to the right of 'site'
        virtual site_type get_nearest_neighbour(site_type site) 
        const 
        override
        {
            site_type nei = site + 1;
            if(nei >= this->L){
                if(this->boundary_condition) nei = nei % L;
                else nei = -1;
            }
            return nei;
        }

        /// @brief Get next-nearest neighbour to given input site (to the right of input site)
        /// @param site site index
        /// @return next-nearest-neighbour to the right of 'site'
        virtual site_type get_next_nearest_neighbour(site_type site) 
        const 
        override
        {
            site_type nei = site + 2;
            if(nei > this->L - 2){
                if(this->boundary_condition) nei = nei % L;
                else nei = -1;
            }
            return nei;
        }

        /// @brief Get all nearest neighbour to given input site (in all directions)
        /// @param site site index
        /// @return all nearest neighbours to the right of 'site'
        virtual std::vector<site_type> get_neighbours(site_type site) 
        const 
        override
        {
            std::vector<site_type> neis;
            
            site_type nei = site + 1;
            if(nei > this->L - 1){
                if(this->boundary_condition) nei = nei % L;
                else nei = -1;
            }
            neis.push_back(nei);
            
            nei = site - 1;
            if(nei < 0){
                if(this->boundary_condition) {
                    nei = (nei % L) + this->L;
                }
                else nei = -1;
            }
            neis.push_back(nei);
            return neis;
        }
    };


}