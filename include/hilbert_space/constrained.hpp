#pragma once

#ifndef _HILBERT_BASE
    #include "_base.hpp"
#endif

namespace QHS{
    /// @brief Hilbert space creator with constraints (i.e. PXP)
    /// @tparam U1_sym choose U(1) symmetry: spin, charge, ...

    typedef std::function<bool(unsigned long long)> _constrained_type;
    
    // template <_constrained_type... constraint_kernels>
    class constrained_hilbert_space : public hilbert_space_base
    {
        _constrained_type constraints;
        /// @brief Initialize hilbert space with given symmetry sector
        /// @tparam U1_sym What kind of U(1) symmetry> charge, spin, etc?
        /// @tparam spinless If chosen U1_sym==charge, are the fermions spinless?
        virtual void init() override {
            //<! create basis for given sector
            this->create_basis();
        }
        
        /// @brief Check if element is allowed under U(1) symmetry
        /// @param idx element to be checked
        /// @return true or false whether element is allowed
        bool check_if_allowed_element(u64 idx)
            { return this->constraints(idx);}
    public:
        constrained_hilbert_space() = default;
        constrained_hilbert_space(int L, _constrained_type&& _constraints)
        { 
            this->system_size = L; 
            this->constraints = _constraints;
            this->init();
        }

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
            
            // for(auto& idx : mapping)
            //     std::cout << idx << "";
            // std::cout << std::endl;
            this->dim = this->mapping.size();
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