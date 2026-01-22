#pragma once

#ifndef _HILBERT_BASE
    #include "_base.hpp"
#endif

namespace QHS{
    /// @brief Hilbert space creator with U(1) symmetry, either spin or charge
    /// @tparam boolean value: spinless fermions?  (valid if chosen U1 == charge)
    /// @tparam U1_sym choose U(1) symmetry: spin, charge, ...
    template <U1 U1_sym = U1::spin, bool spinless = true>
    class U1_hilbert_space : public hilbert_space_base
    {
        protected:

        float U1_sector;
        float max_sector;
        float min_sector;

        /// @brief Initialize hilbert space with given symmetry sector
        /// @tparam U1_sym What kind of U(1) symmetry> charge, spin, etc?
        /// @tparam spinless If chosen U1_sym==charge, are the fermions spinless?
        virtual void init() override {
            if constexpr (U1_sym == U1::spin){
                this->max_sector = this->system_size / 2.;
                this->min_sector = -(int)this->system_size / 2.;
            } else if constexpr (U1_sym == U1::charge){
                this->max_sector = (spinless? 1 : 2) * this->system_size;
                this->min_sector = 0;
            } else {
                this->max_sector = 0;
                this-min_sector = 0;
            }
            bool allowed_sector = (this->U1_sector <= this->max_sector && this->U1_sector >= this->min_sector);
            
            if(allowed_sector == false)
                std::cout << "U(1) sector check:\t\t" << this->U1_sector << "\t\t" << this->min_sector << "\t\t" << this->max_sector << "\t\t" << std::endl;

            _assert_( (allowed_sector == true), NOT_ALLOWED_SYMETRY_SECTOR);

            //<! create basis for given sector
            this->create_basis();
        }
        
        /// @brief Check if element is allowed under U(1) symmetry
        /// @param idx element to be checked
        /// @return true or false whether element is allowed
        virtual bool check_if_allowed_element(u64 idx)
        {
            if constexpr (U1_sym == U1::spin)
                return __builtin_popcountll(idx) == this->U1_sector + this->system_size * _Spin;
            else if constexpr (U1_sym == U1::charge)
                return __builtin_popcountll(idx) == (int)this->U1_sector;
            else
                return idx;
        }
    public:
        U1_hilbert_space() = default;

        /// @brief Constructor for creating Hilbert-space with fixed particle number
        /// @param L Total system size
        /// @param sector  Number of particles (spin ups)
        U1_hilbert_space(int L, float sector = 0)
        { 
            this->system_size = L; 
            this->U1_sector = sector; 
            CONSTRUCTOR_CALL;
			_extra_debug(
				std::cout << FUN_SIGNATURE << "::\n\tHilbert-space initialized with: "
					<< var_name_value(this->system_size, 0) << "\t" 
					<< var_name_value(this->U1_sector, 0) << std::endl;
			)

            this->init();
        }

        auto get_U1_params() { return std::make_pair(this->system_size, this->U1_sector); }

        //<! -------------------------------------------------------- OVERLOADED OPERATORS
        
        /// @brief Create basis with U(1) symmetry multithreaded
        virtual 
        void create_basis() override
        {   
            auto mapping_kernel = [this](u64 start, u64 stop, std::vector<u64>& map_threaded)
            {
                for (u64 j = start; j < stop; j++)
                    if (check_if_allowed_element(j)){
                        // std::cout << j << "\t\t" << to_binary(j, this->system_size) << std::endl;
                        map_threaded.emplace_back(j);
                    }
                //std::cout << map_threaded << std::endl;
            };
            u64 start = 0, stop = ULLPOW(this->system_size);
            u64 two_powL = BinaryPowers[this->system_size];
            if (num_of_threads == 1)
                mapping_kernel(start, stop, this->mapping);
            else {
                //Threaded
                v_2d<u64> map_threaded(num_of_threads);
                std::vector<std::thread> threads;
                threads.reserve(num_of_threads);
                for (int t = 0; t < num_of_threads; t++) {
                    start = (u64)(two_powL / (double)num_of_threads * t);
                    stop = ((t + 1) == num_of_threads ? two_powL : u64(two_powL / (double)num_of_threads * (double)(t + 1)));
                    map_threaded[t] = v_1d<u64>();
                    threads.emplace_back(mapping_kernel, start, stop, ref(map_threaded[t]));
                }
                for (auto& t : threads) t.join();

                for (auto& t : map_threaded)
                    this->mapping.insert(this->mapping.end(), std::make_move_iterator(t.begin()), std::make_move_iterator(t.end()));
            }
            
            this->dim = this->mapping.size();
            
            _extra_debug(
                for(u64 elem : this->mapping)
                    printSeparated(std::cout, "\t", 20, true, elem, boost::dynamic_bitset<>(this->system_size, elem));
                std::cout << "Hilbert-space size = " << this->dim << std::endl;
            );
        }

        /// @brief Overloaded operator to access elements in hilbert space
        /// @param idx Index of element in hilbert space
        /// @return Element of hilbert space at position 'index'
        virtual
        u64 operator()(u64 idx) const override
            { _assert_((idx < this->dim), OUT_OF_MAP);
                return this->mapping[idx]; }


        /// @brief Find index of element in hilbert space
        /// @param element element to find its index
        /// @return index of element 'element'
        virtual 
        u64 find(u64 element) const override
            { return binary_search(this->mapping, 0, this->dim - 1, element); }
    };


    /// @brief Hilbert space creator with U(1) symmetry, either spin or charge
    /// @tparam boolean value: spinless fermions?  (valid if chosen U1 == charge)
    /// @tparam U1_sym choose U(1) symmetry: spin, charge, ...
    template <U1 U1_sym = U1::spin, bool spinless = true>
    class U1_subsystem_hilbert_space : public U1_hilbert_space<U1_sym, spinless>
    {
        protected:
        float U1A_sector;
        float U1B_sector;

        float maxA_sector = 0;
        float minA_sector = 0;
        float maxB_sector = 0;
        float minB_sector = 0;

        int LA;
        int LB;

        u64 maskA;
        u64 maskB;

        /// @brief Initialize hilbert space with given symmetry sector
        /// @tparam U1_sym What kind of U(1) symmetry> charge, spin, etc?
        /// @tparam spinless If chosen U1_sym==charge, are the fermions spinless?
        virtual void init() override {
            if constexpr (U1_sym == U1::spin){
                this->max_sector = this->system_size / 2.;
                this->min_sector = -(int)this->system_size / 2.;
                this->maxA_sector = this->LA / 2.;
                this->minA_sector = -(int)this->LA / 2.;
                this->maxB_sector = this->LB / 2.;
                this->minB_sector = -(int)this->LB / 2.;
            } else if constexpr (U1_sym == U1::charge){
                this->max_sector = (spinless? 1 : 2) * this->system_size;
                this->min_sector = 0;
                this->maxA_sector = (spinless? 1 : 2) * this->LA;
                this->minA_sector = 0;
                this->maxB_sector = (spinless? 1 : 2) * this->LB;
                this->minB_sector = 0;
            } else {
                this->max_sector = 0;
                this->min_sector = 0;
            }
            
            // check if both subsystems have same amount of particles
            bool allowed_sector = (this->U1_sector <= this->max_sector && this->U1_sector >= this->min_sector\
                                    && this->U1A_sector <= this->maxA_sector && this->U1A_sector >= this->minA_sector\
                                    && this->U1B_sector <= this->maxB_sector && this->U1B_sector >= this->minB_sector);
            
            if(allowed_sector == false){
                std::cout << "U(1) sector check:\t\t" << this->U1_sector << "\t\t min: " << this->min_sector << "\t\t max: " << this->max_sector << "\t\t" << std::endl;
                std::cout << "U_A(1) sector check:\t\t" << this->U1A_sector << "\t\t min: " << this->minA_sector << "\t\t max: " << this->maxA_sector << "\t\t" << std::endl;
                std::cout << "U_B(1) sector check:\t\t" << this->U1B_sector << "\t\t min: " << this->minB_sector << "\t\t max: " << this->maxB_sector << "\t\t" << std::endl;
            }

            _assert_( (allowed_sector == true), NOT_ALLOWED_SYMETRY_SECTOR);

            //<! create basis for given sector
            this->create_basis();
        }
        
        /// @brief Check if element is allowed under U(1) symmetry
        /// @param idx element to be checked
        /// @return true or false whether element is allowed
        bool check_if_allowed_element(u64 idx) override
        {
            if constexpr (U1_sym == U1::spin){
                // return __builtin_popcountll(idx) == this->U1_sector + this->system_size * _Spin;
                bool _subA_check = (__builtin_popcountll(idx & maskA) == this->U1A_sector + this->LA * _Spin);
                bool _full_check = (__builtin_popcountll(idx) == this->U1_sector + this->system_size * _Spin);
                return _subA_check && _full_check;
            } else if constexpr (U1_sym == U1::charge){
                // return __builtin_popcountll(idx) == (int)this->U1_sector;
                bool _subA_check = (__builtin_popcountll(idx & maskA) == (int)this->U1A_sector);
                bool _full_check = (__builtin_popcountll(idx) == (int)this->U1_sector);
                return _subA_check && _full_check;
            } else
                return idx;
        }
    public:
        U1_subsystem_hilbert_space() = default;

        /// @brief Constructor for creating Hilbert-space with fixed particle numbers in subsystem A (of size LA) and its complement
        /// @param L Total system size
        /// @param LA Subsystem size
        /// @param NA Number of particles in subsystem A
        /// @param NB Number of particles in subsystem B
        U1_subsystem_hilbert_space(int L, int LA, float NA = 0, float NB = 0)
        { 
            this->system_size = L;
            this->LA = LA;
            this->LB = this->system_size - this->LA;
            this->U1A_sector = NA;
            this->U1B_sector = NB;
            this->U1_sector = NA + NB;

            this->maskA = (ULLPOW(this->LA)-1);
            
            CONSTRUCTOR_CALL;
			_extra_debug(
				std::cout << FUN_SIGNATURE << "::\n\tHilbert-space initialized with: "
					<< var_name_value(this->system_size, 0) 				 << "\t" 
					<< var_name_value(this->LA, 0) << "\t" 
                    << var_name_value(this->LB, 0) << "\t" 
					<< var_name_value(this->U1A_sector, 0) << "\t" 
					<< var_name_value(this->U1B_sector, 0) << "\t" 
					<< var_name_value(this->U1_sector, 0) << "\t" 
					<< var_name_value(this->maskA, 0) << std::endl;
			)

            this->init();
        }

        auto get_U1sub_params() { return std::make_pair(this->system_size, this->U1_sector, this->U1A_sector, this->U1B_sector); }
    };
}