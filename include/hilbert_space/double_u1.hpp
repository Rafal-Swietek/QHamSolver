#pragma once

#ifndef _HILBERT_BASE
    #include "_base.hpp"
#endif

namespace QHS{
    /// @brief Hilbert space creator with U(1) symmetry, either spin or charge
    /// @tparam boolean value: spinless fermions?  (valid if chosen U1 == charge)
    /// @tparam U1_sym choose U(1) symmetry: spin, charge, ...
    template <U1 U1_sym = U1::spin, bool spinless = true>
    class U1U1_hilbert_space : public hilbert_space_base
    {
        protected:

        float U1a_sector;
        float U1b_sector;

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
                this->min_sector = 0;
            }
            bool allowed_sector = (this->U1a_sector <= this->max_sector && this->U1a_sector >= this->min_sector);
            allowed_sector = allowed_sector && (this->U1b_sector <= this->max_sector && this->U1b_sector >= this->min_sector);
            
            if(allowed_sector == false){
                std::cout << "First U(1) sector check:\t\t" << this->U1a_sector << "\t\t" << this->min_sector << "\t\t" << this->max_sector << "\t\t" << std::endl;
                std::cout << "Second U(1) sector check:\t\t" << this->U1b_sector << "\t\t" << this->min_sector << "\t\t" << this->max_sector << "\t\t" << std::endl;
            }

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
                return (countEvenBits(idx) == this->U1a_sector + this->system_size * _Spin)
                    && (countOddBits(idx) == this->U1b_sector + this->system_size * _Spin);
            else if constexpr (U1_sym == U1::charge)
                return (countEvenBits(idx) == (int)this->U1a_sector) && (countOddBits(idx) == (int)this->U1b_sector);
            else
                return idx;
        }

        static
        void generate_sector_states(int pos, int remaining, u64 current_state, std::vector<u64>& states, bool even_side, int L)
        {
            if (remaining == 0) {
                states.emplace_back(current_state);
                return;
            }

            if (pos >= L || remaining > L - pos)
                return;

            const u64 bit = 1ULL << (2 * pos + (even_side ? 0 : 1));
            generate_sector_states(pos + 1, remaining - 1, current_state | bit, states, even_side, L);
            generate_sector_states(pos + 1, remaining, current_state, states, even_side, L);
        }
    public:
        U1U1_hilbert_space() = default;

        /// @brief Constructor for creating Hilbert-space with fixed particle number
        /// @param L Total system size
        /// @param sector  Number of particles (spin ups)
        U1U1_hilbert_space(int L, float sector1 = 0, float sector2 = 0)
        { 
            this->system_size = L; 
            this->U1a_sector = sector1; 
            this->U1b_sector = sector2;
            CONSTRUCTOR_CALL;
			_extra_debug(
				std::cout << FUN_SIGNATURE << "::\n\tHilbert-space initialized with: "
					<< var_name_value(this->system_size, 0) << "\t" 
					<< var_name_value(this->U1a_sector, 0) << "\t" 
					<< var_name_value(this->U1b_sector, 0) << std::endl;
			)

            this->init();
        }

        auto get_U1_params() { return std::make_pair(this->system_size, this->U1a_sector, this->U1b_sector); }

        //<! -------------------------------------------------------- OVERLOADED OPERATORS
        
        /// @brief Create basis with U(1) symmetry using direct U1A/U1B state construction
        virtual 
        void create_basis() override
        {
            const int L = this->system_size;
            int countA = 0;
            int countB = 0;
            if constexpr (U1_sym == U1::spin) {
                countA = (int)(this->U1a_sector + this->system_size * _Spin);
                countB = (int)(this->U1b_sector + this->system_size * _Spin);
            } else if constexpr (U1_sym == U1::charge) {
                countA = (int)this->U1a_sector;
                countB = (int)this->U1b_sector;
            }

            std::vector<u64> even_states;
            std::vector<u64> odd_states;

            generate_sector_states(0, countA, 0ULL, even_states, true, L);
            generate_sector_states(0, countB, 0ULL, odd_states, false, L);

            this->mapping.clear();
            this->mapping.reserve((u64)even_states.size() * (u64)odd_states.size());

            for (u64 even : even_states)
                for (u64 odd : odd_states)
                    this->mapping.emplace_back(even | odd);

            std::sort(this->mapping.begin(), this->mapping.end());
            this->mapping.erase(std::unique(this->mapping.begin(), this->mapping.end()), this->mapping.end());
            this->dim = this->mapping.size();

            _extra_debug(
                for(u64 elem : this->mapping)
                    printSeparated(std::cout, "\t", 20, true, elem, boost::dynamic_bitset<>(block_size * this->system_size, elem));
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
}

// #pragma once

// #ifndef _HILBERT_BASE
//     #include "_base.hpp"
// #endif

// namespace QHS{
//     /// @brief Hilbert space creator with U(1) symmetry, either spin or charge
//     /// @tparam boolean value: spinless fermions?  (valid if chosen U1 == charge)
//     /// @tparam U1_sym choose U(1) symmetry: spin, charge, ...
//     template <U1 U1_sym = U1::spin, bool spinless = true>
//     class U1U1_hilbert_space : public hilbert_space_base
//     {
//         protected:

//         float U1a_sector;
//         float U1b_sector;

//         float max_sector;
//         float min_sector;

//         /// @brief Initialize hilbert space with given symmetry sector
//         /// @tparam U1_sym What kind of U(1) symmetry> charge, spin, etc?
//         /// @tparam spinless If chosen U1_sym==charge, are the fermions spinless?
//         virtual void init() override {
//             if constexpr (U1_sym == U1::spin){
//                 this->max_sector = this->system_size / 2.;
//                 this->min_sector = -(int)this->system_size / 2.;
//             } else if constexpr (U1_sym == U1::charge){
//                 this->max_sector = (spinless? 1 : 2) * this->system_size;
//                 this->min_sector = 0;
//             } else {
//                 this->max_sector = 0;
//                 this-min_sector = 0;
//             }
//             bool allowed_sector = (this->U1a_sector <= this->max_sector && this->U1a_sector >= this->min_sector);
//             allowed_sector = allowed_sector && (this->U1b_sector <= this->max_sector && this->U1b_sector >= this->min_sector);
            
//             if(allowed_sector == false){
//                 std::cout << "First U(1) sector check:\t\t" << this->U1a_sector << "\t\t" << this->min_sector << "\t\t" << this->max_sector << "\t\t" << std::endl;
//                 std::cout << "Second U(1) sector check:\t\t" << this->U1b_sector << "\t\t" << this->min_sector << "\t\t" << this->max_sector << "\t\t" << std::endl;
//             }

//             _assert_( (allowed_sector == true), NOT_ALLOWED_SYMETRY_SECTOR);

//             //<! create basis for given sector
//             this->create_basis();
//         }
        
//         /// @brief Check if element is allowed under U(1) symmetry
//         /// @param idx element to be checked
//         /// @return true or false whether element is allowed
//         virtual bool check_if_allowed_element(u64 idx)
//         {
//             if constexpr (U1_sym == U1::spin)
//                 return (countEvenBits(idx) == this->U1a_sector + this->system_size * _Spin)
//                     && (countOddBits(idx) == this->U1b_sector + this->system_size * _Spin);
//             else if constexpr (U1_sym == U1::charge)
//                 return (countEvenBits(idx) == (int)this->U1a_sector) && (countOddBits(idx) == (int)this->U1b_sector);
//             else
//                 return idx;
//         }
//     public:
//         U1U1_hilbert_space() = default;

//         /// @brief Constructor for creating Hilbert-space with fixed particle number
//         /// @param L Total system size
//         /// @param sector  Number of particles (spin ups)
//         U1U1_hilbert_space(int L, float sector1 = 0, float sector2 = 0)
//         { 
//             this->system_size = L; 
//             this->U1a_sector = sector1; 
//             this->U1b_sector = sector2;
//             CONSTRUCTOR_CALL;
// 			_extra_debug(
// 				std::cout << FUN_SIGNATURE << "::\n\tHilbert-space initialized with: "
// 					<< var_name_value(this->system_size, 0) << "\t" 
// 					<< var_name_value(this->U1a_sector, 0) << "\t" 
// 					<< var_name_value(this->U1b_sector, 0) << std::endl;
// 			)

//             this->init();
//         }

//         auto get_U1_params() { return std::make_pair(this->system_size, this->U1a_sector, this->U1b_sector); }

//         //<! -------------------------------------------------------- OVERLOADED OPERATORS
        
//         /// @brief Create basis with U(1) symmetry multithreaded
//         virtual 
//         void create_basis() override
//         {   
//             auto mapping_kernel = [this](u64 start, u64 stop, std::vector<u64>& map_threaded)
//             {
//                 for (u64 j = start; j < stop; j++)
//                     if (check_if_allowed_element(j)){
//                         // std::cout << j << "\t\t" << to_binary(j, this->system_size) << std::endl;
//                         map_threaded.emplace_back(j);
//                     }
//                 //std::cout << map_threaded << std::endl;
//             };
//             u64 start = 0, stop = BinaryPowers[2*this->system_size];
//             u64 four_powL = BinaryPowers[2*this->system_size];
//             if (num_of_threads == 1)
//                 mapping_kernel(start, stop, this->mapping);
//             else {
//                 //Threaded
//                 v_2d<u64> map_threaded(num_of_threads);
//                 std::vector<std::thread> threads;
//                 threads.reserve(num_of_threads);
//                 for (int t = 0; t < num_of_threads; t++) {
//                     start = (u64)(four_powL / (double)num_of_threads * t);
//                     stop = ((t + 1) == num_of_threads ? four_powL : u64(four_powL / (double)num_of_threads * (double)(t + 1)));
//                     map_threaded[t] = v_1d<u64>();
//                     threads.emplace_back(mapping_kernel, start, stop, ref(map_threaded[t]));
//                 }
//                 for (auto& t : threads) t.join();

//                 for (auto& t : map_threaded)
//                     this->mapping.insert(this->mapping.end(), std::make_move_iterator(t.begin()), std::make_move_iterator(t.end()));
//             }
            
//             this->dim = this->mapping.size();
            
//             _extra_debug(
//                 for(u64 elem : this->mapping)
//                     printSeparated(std::cout, "\t", 20, true, elem, boost::dynamic_bitset<>(block_size * this->system_size, elem));
//                 std::cout << "Hilbert-space size = " << this->dim << std::endl;
//             );
//         }

//         /// @brief Overloaded operator to access elements in hilbert space
//         /// @param idx Index of element in hilbert space
//         /// @return Element of hilbert space at position 'index'
//         virtual
//         u64 operator()(u64 idx) const override
//             { _assert_((idx < this->dim), OUT_OF_MAP);
//                 return this->mapping[idx]; }


//         /// @brief Find index of element in hilbert space
//         /// @param element element to find its index
//         /// @return index of element 'element'
//         virtual 
//         u64 find(u64 element) const override
//             { return binary_search(this->mapping, 0, this->dim - 1, element); }
//     };
// }