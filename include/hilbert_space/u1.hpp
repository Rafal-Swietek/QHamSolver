#pragma once

#ifndef _HILBERT_BASE
    #include "_base.hpp"
#endif

//<! Enum for possible U(1) symmetries: for now charge and spin
enum class U1 {spin, charge};

/// @brief Hilbert space creator with U(1) symmetry, either spin or charge
/// @tparam boolean value: spinless fermions?  (valid if chosen U1 == charge)
/// @tparam U1_sym choose U(1) symmetry: spin, charge, ...
template <U1 U1_sym = U1::spin, bool spinless = true>
class U1_hilbert_space : public hilbert_space_base
{
    float U1_sector;
    float max_sector;
    float min_sector;

    /// @brief Initialize hilbert space with given symmetry sector
    /// @tparam U1_sym What kind of U(1) symmetry> charge, spin, etc?
    /// @tparam spinless If chosen U1_sym==charge, are the fermions spinless?
    virtual void init() override {
        
        //<! check if input sector is allowed by system
        //switch(U1_sym){
        //case U1::spin :
        //    this->max_sector = this->system_size / 2.;
        //    this->min_sector = -this->system_size / 2.;
        //case U1::charge :
        //    this->max_sector = (spinless? 1 : 2) * this->system_size;
        //    this->min_sector = 0;
        //}
        if constexpr (U1_sym == U1::spin){
            this->max_sector = this->system_size / 2.;
            this->min_sector = -this->system_size / 2.;
        } else if constexpr (U1_sym == U1::spin){
            this->max_sector = (spinless? 1 : 2) * this->system_size;
            this->min_sector = 0;
        } else {
            this->max_sector = 0;
            this-min_sector = 0;
        }
        bool allowed_sector = (this->U1_sector < this->max_sector && this->U1_sector > this->min_sector);
        
        if(allowed_sector == false)
            std::cout << this->U1_sector << "\t\t" << this->min_sector << "\t\t" << this->max_sector << "\t\t" << std::endl;

        _assert_( (allowed_sector == false), NOT_ALLOWED_SYMETRY_SECTOR);

        //<! create basis for given sector
        this->create_basis();
    }
    
    /// @brief Check if element is allowed under U(1) symmetry
    /// @param idx element to be checked
    /// @return true or false whether element is allowed
    bool check_if_allowed_element(u64 idx)
    {
        //switch(U1_sym){
        //case U1::spin :    return __builtin_popcountll(idx) - this->system_size / 2. == this->U1_sector;
        //case U1::charge :  return __builtin_popcountll(idx) == this->U1_sector;
        //default:
        //    return __builtin_popcountll(idx) - this->system_size / 2. == this->U1_sector;
        //}
        if constexpr (U1_sym == U1::spin)
            return __builtin_popcountll(idx) - this->system_size / 2. == this->U1_sector;
        else if constexpr (U1_sym == U1::spin)
            return __builtin_popcountll(idx) == this->U1_sector;
        else
            return idx;
    }
public:
    U1_hilbert_space() = default;
    U1_hilbert_space(unsigned int L, float sector = 0)
    { 
        this->system_size = L; 
        this->U1_sector = sector; 
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
                if (check_if_allowed_element(j)) 
                    map_threaded.emplace_back(j);
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