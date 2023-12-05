#pragma once

namespace lattice{

    class lattice2D : public lattice_base{
    private:
        site_type Lx;
        site_type Ly;

        site_type A;    //<! surface area of system
    public:

        //---------------------------------------------- CONSTRUCTORS
        lattice2D() = default;
        
        //<! general constructor(rectangles)
        lattice2D(site_type _Lx, site_type _Ly, bool _bound_cond = true)
        {
            this->Lx = _Lx;
            this->Ly = _Ly;
            this->boundary_condition = _bound_cond;
            this->A = this->Lx * this->Ly;
            this->volume = this->A;
        };

        //<! square constructor(rectangles)
        lattice2D(site_type _L, bool _bound_cond = true)
        {
            this->Lx = _L;
            this->Ly = _L;
            this->boundary_condition = _bound_cond;
            this->A = this->Lx * this->Ly;
            this->volume = this->A;
        };

        //---------------------------------------------- NEIGHBOURS
        /// @brief Get nearest neighbour to given input site (to the right of input site)
        /// @param site site index
        /// @return nearest-neighbour to the right of 'site'
        virtual site_type get_nearest_neighbour(site_type site) 
        const 
        override
        {
            // go right (x++, y)
            if(site % this->Lx == this->Lx - 1){
                if(this->boundary_condition) return site + 1 - this->Lx;
                else return -1;
            }
            else 
                return site + 1;
        }

        /// @brief Get next-nearest neighbour to given input site (to the right of input site)
        /// @param site site index
        /// @return next-nearest-neighbour to the right of 'site'
        virtual site_type get_next_nearest_neighbour(site_type site) 
        const 
        override
        {
            // go right-up (x++, y++)
            if(site % this->volume == this->volume - 1){
                if(this->boundary_condition) return 0;
                else return -1;
            }
            else if(site % this->volume >= this->volume - this->Lx){     // move across Y boundary
                if(this->boundary_condition) return site + 1 - this->Lx * (this->Ly - 1);
                else return -1;
            }
            else 
                return site + 1;                                         // includes moving across X boundary
        }

        /// @brief Get all nearest neighbour to given input site (in all directions)
        /// @param site site index
        /// @return all nearest neighbours to the right of 'site'
        virtual std::vector<site_type> get_neighbours(site_type site) 
        const 
        override
        {
            std::vector<site_type> neis;
            site_type nei;

            // go right along X
            if(site % this->Lx == this->Lx - 1){
                if(this->boundary_condition) nei = site + 1 - this->Lx;
                else nei = -1;
            }
            else 
                nei = site + 1;
            neis.push_back(nei);
            
            // go left along X
            if(site % this->Lx == 0){
                if(this->boundary_condition) nei = site - 1 + this->Lx;
                else nei = -1;
            }
            else 
                nei = site - 1;
            neis.push_back(nei);
            
            // go up along Y
            if(site >= this->A - this->Lx ){
                if(this->boundary_condition) nei = site % this->Lx;
                else nei = -1;
            }
            else 
                nei = site + this->Lx;
            neis.push_back(nei);
            
            // go down along Y
            if(site < this->Lx){
                if(this->boundary_condition) nei = this->A - this->Lx + site;
                else nei = -1;
            }
            else 
                nei = site - this->Lx;
            neis.push_back(nei);
            return neis;
            
        }
    };

}