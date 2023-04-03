#pragma once

#define NOT_ALLOWED_SYMETRY_SECTOR  "ERROR 1: Chosen symmetry sector not allowed, exceeds min/max values;"
#define OUT_OF_MAP                  "ERROR 2: Given index is not included in hilbert space"

#ifndef _HILBERT_BASE
#define _HILBERT_BASE

/// @brief Base class for hilbert space construction
/// @tparam ...constraints 
//template <typename... constraints>
class hilbert_space_base {
    
    protected:
        //variadic_struct<constraints...> sectors;

        std::vector<u64> mapping;
        unsigned int system_size;
        u64 dim;
        virtual void init() = 0;
    public:
        virtual ~hilbert_space_base() = default;
        auto get_hilbert_space_size() const { return this->dim; }
        auto get_mapping() const { return this->mapping; }
        virtual void create_basis() = 0;
        
        virtual u64 operator()(u64 idx) const = 0;
        virtual u64 find(u64 idx)       const = 0;
};

//hilbert_space_base::~hilbert_space_base(){}



/// @brief Hilbert space with no symmetries
class full_hilbert_space : public hilbert_space_base{
    
    //<! Someday might need to add stuff..
    virtual void init() override 
        {};
    public:
        full_hilbert_space() = default;
        full_hilbert_space(unsigned int L)
        { 
            this->system_size = L; 
            this->dim = ULLPOW(L);
            this->init();
        }
        
        virtual u64 operator()(u64 idx) const override { return idx; };
        virtual u64 find(u64 idx)       const override { return idx; };
        virtual void create_basis() override 
            { std::cout << "AIN'T DO NOTHING! Hilbert space is created as full." << std::endl; }
};

#endif