#pragma once

#define NOT_ALLOWED_SYMETRY_SECTOR  "ERROR 1: Chosen symmetry sector not allowed, exceeds min/max values;"
#define OUT_OF_MAP                  "ERROR 2: Given index is not included in hilbert space"

#ifndef _HILBERT_BASE
#define _HILBERT_BASE

namespace QHS{

    /// @tparam ...constraints 
    //template <typename... constraints>

    /// @brief Base class for hilbert space construction
    class hilbert_space_base {
        
        protected:
            //variadic_struct<constraints...> sectors;
            using iterator          = std::vector<u64>::iterator;
            using const_iterator    = std::vector<u64>::const_iterator;
            using ptr               = u64*;
            using const_ptr         = u64 const*;

            std::vector<u64> mapping;
            int system_size;
            u64 dim;
            virtual void init() = 0;
        public:
            virtual ~hilbert_space_base() = default;
            auto get_hilbert_space_size() const { return this->dim; }
            auto get_mapping() const { return this->mapping; }
            virtual void create_basis() = 0;
            
            auto set_mapping(std::vector<u64> input_map) 
                { this->mapping = input_map; }

            virtual u64 operator()(u64 idx) const = 0;
            virtual u64 find(u64 idx)       const = 0;

            //<! ------------------------------------------ ITERATORS FOR RANGE_BASED LOOPS
            virtual iterator begin()              { return this->mapping.begin(); }
            virtual iterator end()                { return this->mapping.end(); }
            virtual const_iterator cbegin() const { return this->mapping.begin(); }
            virtual const_iterator cend()   const { return this->mapping.end(); }
            virtual const_iterator begin()  const { return this->mapping.begin(); }
            virtual const_iterator end()    const { return this->mapping.end(); }

            //<! ------------------------------------------ ITERATORS FOR NOT CONTROLLED INSTANCE
            friend ptr begin(hilbert_space_base& _hilbert_space)  { return _hilbert_space.mapping.data(); }
            friend ptr end(  hilbert_space_base& _hilbert_space)  { return _hilbert_space.mapping.data() + _hilbert_space.dim; }
            
            friend const_ptr cbegin(hilbert_space_base const& _hilbert_space) { return _hilbert_space.mapping.data(); }
            friend const_ptr cend(  hilbert_space_base const& _hilbert_space) { return _hilbert_space.mapping.data() + _hilbert_space.dim; }
            friend const_ptr begin( hilbert_space_base const& _hilbert_space) { return _hilbert_space.mapping.data(); }
            friend const_ptr end(   hilbert_space_base const& _hilbert_space) { return _hilbert_space.mapping.data() + _hilbert_space.dim; }
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

}
#endif