#if !defined(_HASH_H)
#define _HASH_H

/*
 This code is adapted from the khash library, version 0.2.8.
 It has been modified to fit into PETSc.
 Original copyright notice follows.
*/

/* The MIT License

   Copyright (c) 2008, 2009, 2011 by Attractive Chaos <attractor@live.co.uk>

   Permission is hereby granted, free of charge, to any person obtaining
   a copy of this software and associated documentation files (the
   "Software"), to deal in the Software without restriction, including
   without limitation the rights to use, copy, modify, merge, publish,
   distribute, sublicense, and/or sell copies of the Software, and to
   permit persons to whom the Software is furnished to do so, subject to
   the following conditions:

   The above copyright notice and this permission notice shall be
   included in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
   EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
   MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
   NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
   BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
   ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
   CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
   SOFTWARE.
*/

/*
  An example:

#include "khash.h"
KHASH_MAP_INIT_INT(32, char)
int main() {
	int ret, is_missing;
	khiter_t k;
	khash_t(32) *h = kh_init(32);
	k = kh_put(32, h, 5, &ret);
	kh_value(h, k) = 10;
	k = kh_get(32, h, 10);
	is_missing = (k == kh_end(h));
	k = kh_get(32, h, 5);
	kh_del(32, h, k);
	for (k = kh_begin(h); k != kh_end(h); ++k)
		if (kh_exist(h, k)) kh_value(h, k) = 1;
	kh_destroy(32, h);
	return 0;
}
*/

/*
  2013-05-02 (0.2.8):

	* Use quadratic probing. When the capacity is power of 2, stepping function
	  i*(i+1)/2 guarantees to traverse each bucket. It is better than double
	  hashing on cache performance and is more robust than linear probing.

	  In theory, double hashing should be more robust than quadratic probing.
	  However, my implementation is probably not for large hash tables, because
	  the second hash function is closely tied to the first hash function,
	  which reduce the effectiveness of double hashing.

	Reference: http://research.cs.vt.edu/AVresearch/hashing/quadratic.php

  2011-12-29 (0.2.7):

    * Minor code clean up; no actual effect.

  2011-09-16 (0.2.6):

	* The capacity is a power of 2. This seems to dramatically improve the
	  speed for simple keys. Thank Zilong Tan for the suggestion. Reference:

	   - http://code.google.com/p/ulib/
	   - http://nothings.org/computer/judy/

	* Allow to optionally use linear probing which usually has better
	  performance for random input. Double hashing is still the default as it
	  is more robust to certain non-random input.

	* Added Wang's integer hash function (not used by default). This hash
	  function is more robust to certain non-random input.

  2011-02-14 (0.2.5):

    * Allow to declare global functions.

  2009-09-26 (0.2.4):

    * Improve portability

  2008-09-19 (0.2.3):

	* Corrected the example
	* Improved interfaces

  2008-09-11 (0.2.2):

	* Improved speed a little in kh_put()

  2008-09-10 (0.2.1):

	* Added kh_clear()
	* Fixed a compiling error

  2008-09-02 (0.2.0):

	* Changed to token concatenation which increases flexibility.

  2008-08-31 (0.1.2):

	* Fixed a bug in kh_get(), which has not been tested previously.

  2008-08-31 (0.1.1):

	* Added destructor
*/


#ifndef __AC_KHASH_H
#define __AC_KHASH_H

/*!
  @header

  Generic hash table library.
 */

#define AC_VERSION_KHASH_H "0.2.8"

#include <stdlib.h>
#include <string.h>
#include <limits.h>

/* compiler specific configuration */

#if UINT_MAX == 0xffffffffu
typedef unsigned int khint32_t;
#elif ULONG_MAX == 0xffffffffu
typedef unsigned long khint32_t;
#endif

#if ULONG_MAX == ULLONG_MAX
typedef unsigned long khint64_t;
#else
typedef unsigned long long khint64_t;
#endif

#ifndef kh_inline
#ifdef _MSC_VER
#define kh_inline __inline
#else
#define kh_inline inline
#endif
#endif /* kh_inline */

#ifndef klib_unused
#if (defined __clang__ && __clang_major__ >= 3) || (defined __GNUC__ && __GNUC__ >= 3)
#define klib_unused __attribute__ ((__unused__))
#else
#define klib_unused
#endif
#endif /* klib_unused */

typedef khint32_t khint_t;
typedef khint_t khiter_t;

#define __ac_isempty(flag, i) ((flag[i>>4]>>((i&0xfU)<<1))&2)
#define __ac_isdel(flag, i) ((flag[i>>4]>>((i&0xfU)<<1))&1)
#define __ac_iseither(flag, i) ((flag[i>>4]>>((i&0xfU)<<1))&3)
#define __ac_set_isdel_false(flag, i) (flag[i>>4]&=~(1ul<<((i&0xfU)<<1)))
#define __ac_set_isempty_false(flag, i) (flag[i>>4]&=~(2ul<<((i&0xfU)<<1)))
#define __ac_set_isboth_false(flag, i) (flag[i>>4]&=~(3ul<<((i&0xfU)<<1)))
#define __ac_set_isdel_true(flag, i) (flag[i>>4]|=1ul<<((i&0xfU)<<1))

#define __ac_fsize(m) ((m) < 16? 1 : (m)>>4)

#ifndef kroundup32
#define kroundup32(x) (--(x), (x)|=(x)>>1, (x)|=(x)>>2, (x)|=(x)>>4, (x)|=(x)>>8, (x)|=(x)>>16, ++(x))
#endif

#ifndef kcalloc
#define kcalloc(N,Z) calloc(N,Z)
#endif
#ifndef kmalloc
#define kmalloc(Z) malloc(Z)
#endif
#ifndef krealloc
#define krealloc(P,Z) realloc(P,Z)
#endif
#ifndef kfree
#define kfree(P) free(P)
#endif

static const double __ac_HASH_UPPER = 0.77;

#define __KHASH_TYPE(name, khkey_t, khval_t) \
	typedef struct kh_##name##_s { \
		khint_t n_buckets, size, n_occupied, upper_bound; \
		khint32_t *flags; \
		khkey_t *keys; \
		khval_t *vals; \
	} kh_##name##_t;

#define __KHASH_PROTOTYPES(name, khkey_t, khval_t)	 					\
	extern kh_##name##_t *kh_init_##name(void);							\
	extern void kh_destroy_##name(kh_##name##_t *h);					\
	extern void kh_clear_##name(kh_##name##_t *h);						\
	extern khint_t kh_get_##name(const kh_##name##_t *h, khkey_t key); 	\
	extern int kh_resize_##name(kh_##name##_t *h, khint_t new_n_buckets); \
	extern khint_t kh_put_##name(kh_##name##_t *h, khkey_t key, int *ret); \
	extern void kh_del_##name(kh_##name##_t *h, khint_t x);

#define __KHASH_IMPL(name, SCOPE, khkey_t, khval_t, kh_is_map, __hash_func, __hash_equal) \
	SCOPE kh_##name##_t *kh_init_##name(void) {							\
		return (kh_##name##_t*)kcalloc(1, sizeof(kh_##name##_t));		\
	}																	\
	SCOPE void kh_destroy_##name(kh_##name##_t *h)						\
	{																	\
		if (h) {														\
			kfree((void *)h->keys); kfree(h->flags);					\
			kfree((void *)h->vals);										\
			kfree(h);													\
		}																\
	}																	\
	SCOPE void kh_clear_##name(kh_##name##_t *h)						\
	{																	\
		if (h && h->flags) {											\
			memset(h->flags, 0xaa, __ac_fsize(h->n_buckets) * sizeof(khint32_t)); \
			h->size = h->n_occupied = 0;								\
		}																\
	}																	\
	SCOPE khint_t kh_get_##name(const kh_##name##_t *h, khkey_t key) 	\
	{																	\
		if (h->n_buckets) {												\
			khint_t k, i, last, mask, step = 0; \
			mask = h->n_buckets - 1;									\
			k = __hash_func(key); i = k & mask;							\
			last = i; \
			while (!__ac_isempty(h->flags, i) && (__ac_isdel(h->flags, i) || !__hash_equal(h->keys[i], key))) { \
				i = (i + (++step)) & mask; \
				if (i == last) return h->n_buckets;						\
			}															\
			return __ac_iseither(h->flags, i)? h->n_buckets : i;		\
		} else return 0;												\
	}																	\
	SCOPE int kh_resize_##name(kh_##name##_t *h, khint_t new_n_buckets) \
	{ /* This function uses 0.25*n_buckets bytes of working space instead of [sizeof(key_t+val_t)+.25]*n_buckets. */ \
		khint32_t *new_flags = 0;										\
		khint_t j = 1;													\
		{																\
			kroundup32(new_n_buckets); 									\
			if (new_n_buckets < 4) new_n_buckets = 4;					\
			if (h->size >= (khint_t)(new_n_buckets * __ac_HASH_UPPER + 0.5)) j = 0;	/* requested size is too small */ \
			else { /* hash table size to be changed (shrink or expand); rehash */ \
				new_flags = (khint32_t*)kmalloc(__ac_fsize(new_n_buckets) * sizeof(khint32_t));	\
				if (!new_flags) return -1;								\
				memset(new_flags, 0xaa, __ac_fsize(new_n_buckets) * sizeof(khint32_t)); \
				if (h->n_buckets < new_n_buckets) {	/* expand */		\
					khkey_t *new_keys = (khkey_t*)krealloc((void *)h->keys, new_n_buckets * sizeof(khkey_t)); \
					if (!new_keys) { kfree(new_flags); return -1; }		\
					h->keys = new_keys;									\
					if (kh_is_map) {									\
						khval_t *new_vals = (khval_t*)krealloc((void *)h->vals, new_n_buckets * sizeof(khval_t)); \
						if (!new_vals) { kfree(new_flags); return -1; }	\
						h->vals = new_vals;								\
					}													\
				} /* otherwise shrink */								\
			}															\
		}																\
		if (j) { /* rehashing is needed */								\
			for (j = 0; j != h->n_buckets; ++j) {						\
				if (__ac_iseither(h->flags, j) == 0) {					\
					khkey_t key = h->keys[j];							\
					khval_t val;										\
					khint_t new_mask;									\
					new_mask = new_n_buckets - 1; 						\
					if (kh_is_map) val = h->vals[j];					\
					__ac_set_isdel_true(h->flags, j);					\
					while (1) { /* kick-out process; sort of like in Cuckoo hashing */ \
						khint_t k, i, step = 0; \
						k = __hash_func(key);							\
						i = k & new_mask;								\
						while (!__ac_isempty(new_flags, i)) i = (i + (++step)) & new_mask; \
						__ac_set_isempty_false(new_flags, i);			\
						if (i < h->n_buckets && __ac_iseither(h->flags, i) == 0) { /* kick out the existing element */ \
							{ khkey_t tmp = h->keys[i]; h->keys[i] = key; key = tmp; } \
							if (kh_is_map) { khval_t tmp = h->vals[i]; h->vals[i] = val; val = tmp; } \
							__ac_set_isdel_true(h->flags, i); /* mark it as deleted in the old hash table */ \
						} else { /* write the element and jump out of the loop */ \
							h->keys[i] = key;							\
							if (kh_is_map) h->vals[i] = val;			\
							break;										\
						}												\
					}													\
				}														\
			}															\
			if (h->n_buckets > new_n_buckets) { /* shrink the hash table */ \
				h->keys = (khkey_t*)krealloc((void *)h->keys, new_n_buckets * sizeof(khkey_t)); \
				if (kh_is_map) h->vals = (khval_t*)krealloc((void *)h->vals, new_n_buckets * sizeof(khval_t)); \
			}															\
			kfree(h->flags); /* free the working space */				\
			h->flags = new_flags;										\
			h->n_buckets = new_n_buckets;								\
			h->n_occupied = h->size;									\
			h->upper_bound = (khint_t)(h->n_buckets * __ac_HASH_UPPER + 0.5); \
		}																\
		return 0;														\
	}																	\
	SCOPE khint_t kh_put_##name(kh_##name##_t *h, khkey_t key, int *ret) \
	{																	\
		khint_t x;														\
		if (h->n_occupied >= h->upper_bound) { /* update the hash table */ \
			if (h->n_buckets > (h->size<<1)) {							\
				if (kh_resize_##name(h, h->n_buckets - 1) < 0) { /* clear "deleted" elements */ \
					*ret = -1; return h->n_buckets;						\
				}														\
			} else if (kh_resize_##name(h, h->n_buckets + 1) < 0) { /* expand the hash table */ \
				*ret = -1; return h->n_buckets;							\
			}															\
		} /* TODO: to implement automatically shrinking; resize() already support shrinking */ \
		{																\
			khint_t k, i, site, last, mask = h->n_buckets - 1, step = 0; \
			x = site = h->n_buckets; k = __hash_func(key); i = k & mask; \
			if (__ac_isempty(h->flags, i)) x = i; /* for speed up */	\
			else {														\
				last = i; \
				while (!__ac_isempty(h->flags, i) && (__ac_isdel(h->flags, i) || !__hash_equal(h->keys[i], key))) { \
					if (__ac_isdel(h->flags, i)) site = i;				\
					i = (i + (++step)) & mask; \
					if (i == last) { x = site; break; }					\
				}														\
				if (x == h->n_buckets) {								\
					if (__ac_isempty(h->flags, i) && site != h->n_buckets) x = site; \
					else x = i;											\
				}														\
			}															\
		}																\
		if (__ac_isempty(h->flags, x)) { /* not present at all */		\
			h->keys[x] = key;											\
			__ac_set_isboth_false(h->flags, x);							\
			++h->size; ++h->n_occupied;									\
			*ret = 1;													\
		} else if (__ac_isdel(h->flags, x)) { /* deleted */				\
			h->keys[x] = key;											\
			__ac_set_isboth_false(h->flags, x);							\
			++h->size;													\
			*ret = 2;													\
		} else *ret = 0; /* Don't touch h->keys[x] if present and not deleted */ \
		return x;														\
	}																	\
	SCOPE void kh_del_##name(kh_##name##_t *h, khint_t x)				\
	{																	\
		if (x != h->n_buckets && !__ac_iseither(h->flags, x)) {			\
			__ac_set_isdel_true(h->flags, x);							\
			--h->size;													\
		}																\
	}

#define KHASH_DECLARE(name, khkey_t, khval_t)		 					\
	__KHASH_TYPE(name, khkey_t, khval_t) 								\
	__KHASH_PROTOTYPES(name, khkey_t, khval_t)

#define KHASH_INIT2(name, SCOPE, khkey_t, khval_t, kh_is_map, __hash_func, __hash_equal) \
	__KHASH_TYPE(name, khkey_t, khval_t) 								\
	__KHASH_IMPL(name, SCOPE, khkey_t, khval_t, kh_is_map, __hash_func, __hash_equal)

#define KHASH_INIT(name, khkey_t, khval_t, kh_is_map, __hash_func, __hash_equal) \
	KHASH_INIT2(name, static kh_inline klib_unused, khkey_t, khval_t, kh_is_map, __hash_func, __hash_equal)

/* --- BEGIN OF HASH FUNCTIONS --- */

/*! @function
  @abstract     Integer hash function
  @param  key   The integer [khint32_t]
  @return       The hash value [khint_t]
 */
#define kh_int_hash_func(key) (khint32_t)(key)
/*! @function
  @abstract     Integer comparison function
 */
#define kh_int_hash_equal(a, b) ((a) == (b))
/*! @function
  @abstract     64-bit integer hash function
  @param  key   The integer [khint64_t]
  @return       The hash value [khint_t]
 */
#define kh_int64_hash_func(key) (khint32_t)((key)>>33^(key)^(key)<<11)
/*! @function
  @abstract     64-bit integer comparison function
 */
#define kh_int64_hash_equal(a, b) ((a) == (b))
/*! @function
  @abstract     const char* hash function
  @param  s     Pointer to a null terminated string
  @return       The hash value
 */
static kh_inline khint_t __ac_X31_hash_string(const char *s)
{
	khint_t h = (khint_t)*s;
	if (h) for (++s ; *s; ++s) h = (h << 5) - h + (khint_t)*s;
	return h;
}
/*! @function
  @abstract     Another interface to const char* hash function
  @param  key   Pointer to a null terminated string [const char*]
  @return       The hash value [khint_t]
 */
#define kh_str_hash_func(key) __ac_X31_hash_string(key)
/*! @function
  @abstract     Const char* comparison function
 */
#define kh_str_hash_equal(a, b) (strcmp(a, b) == 0)

static kh_inline khint_t __ac_Wang_hash(khint_t key)
{
    key += ~(key << 15);
    key ^=  (key >> 10);
    key +=  (key << 3);
    key ^=  (key >> 6);
    key += ~(key << 11);
    key ^=  (key >> 16);
    return key;
}
#define kh_int_hash_func2(key) __ac_Wang_hash((khint_t)key)

/* --- END OF HASH FUNCTIONS --- */

/* Other convenient macros... */

/*!
  @abstract Type of the hash table.
  @param  name  Name of the hash table [symbol]
 */
#define khash_t(name) kh_##name##_t

/*! @function
  @abstract     Initiate a hash table.
  @param  name  Name of the hash table [symbol]
  @return       Pointer to the hash table [khash_t(name)*]
 */
#define kh_init(name) kh_init_##name()

/*! @function
  @abstract     Destroy a hash table.
  @param  name  Name of the hash table [symbol]
  @param  h     Pointer to the hash table [khash_t(name)*]
 */
#define kh_destroy(name, h) kh_destroy_##name(h)

/*! @function
  @abstract     Reset a hash table without deallocating memory.
  @param  name  Name of the hash table [symbol]
  @param  h     Pointer to the hash table [khash_t(name)*]
 */
#define kh_clear(name, h) kh_clear_##name(h)

/*! @function
  @abstract     Resize a hash table.
  @param  name  Name of the hash table [symbol]
  @param  h     Pointer to the hash table [khash_t(name)*]
  @param  s     New size [khint_t]
 */
#define kh_resize(name, h, s) kh_resize_##name(h, s)

/*! @function
  @abstract     Insert a key to the hash table.
  @param  name  Name of the hash table [symbol]
  @param  h     Pointer to the hash table [khash_t(name)*]
  @param  k     Key [type of keys]
  @param  r     Extra return code: -1 if the operation failed;
                0 if the key is present in the hash table;
                1 if the bucket is empty (never used); 2 if the element in
				the bucket has been deleted [int*]
  @return       Iterator to the inserted element [khint_t]
 */
#define kh_put(name, h, k, r) kh_put_##name(h, k, r)

/*! @function
  @abstract     Retrieve a key from the hash table.
  @param  name  Name of the hash table [symbol]
  @param  h     Pointer to the hash table [khash_t(name)*]
  @param  k     Key [type of keys]
  @return       Iterator to the found element, or kh_end(h) if the element is absent [khint_t]
 */
#define kh_get(name, h, k) kh_get_##name(h, k)

/*! @function
  @abstract     Remove a key from the hash table.
  @param  name  Name of the hash table [symbol]
  @param  h     Pointer to the hash table [khash_t(name)*]
  @param  k     Iterator to the element to be deleted [khint_t]
 */
#define kh_del(name, h, k) kh_del_##name(h, k)

/*! @function
  @abstract     Test whether a bucket contains data.
  @param  h     Pointer to the hash table [khash_t(name)*]
  @param  x     Iterator to the bucket [khint_t]
  @return       1 if containing data; 0 otherwise [int]
 */
#define kh_exist(h, x) (!__ac_iseither((h)->flags, (x)))

/*! @function
  @abstract     Get key given an iterator
  @param  h     Pointer to the hash table [khash_t(name)*]
  @param  x     Iterator to the bucket [khint_t]
  @return       Key [type of keys]
 */
#define kh_key(h, x) ((h)->keys[x])

/*! @function
  @abstract     Get value given an iterator
  @param  h     Pointer to the hash table [khash_t(name)*]
  @param  x     Iterator to the bucket [khint_t]
  @return       Value [type of values]
  @discussion   For hash sets, calling this results in segfault.
 */
#define kh_val(h, x) ((h)->vals[x])

/*! @function
  @abstract     Alias of kh_val()
 */
#define kh_value(h, x) ((h)->vals[x])

/*! @function
  @abstract     Get the start iterator
  @param  h     Pointer to the hash table [khash_t(name)*]
  @return       The start iterator [khint_t]
 */
#define kh_begin(h) (khint_t)(0)

/*! @function
  @abstract     Get the end iterator
  @param  h     Pointer to the hash table [khash_t(name)*]
  @return       The end iterator [khint_t]
 */
#define kh_end(h) ((h)->n_buckets)

/*! @function
  @abstract     Get the number of elements in the hash table
  @param  h     Pointer to the hash table [khash_t(name)*]
  @return       Number of elements in the hash table [khint_t]
 */
#define kh_size(h) ((h)->size)

/*! @function
  @abstract     Get the number of buckets in the hash table
  @param  h     Pointer to the hash table [khash_t(name)*]
  @return       Number of buckets in the hash table [khint_t]
 */
#define kh_n_buckets(h) ((h)->n_buckets)

/*! @function
  @abstract     Iterate over the entries in the hash table
  @param  h     Pointer to the hash table [khash_t(name)*]
  @param  kvar  Variable to which key will be assigned
  @param  vvar  Variable to which value will be assigned
  @param  code  Block of code to execute
 */
#define kh_foreach(h, kvar, vvar, code) { khint_t __i;		\
	for (__i = kh_begin(h); __i != kh_end(h); ++__i) {		\
		if (!kh_exist(h,__i)) continue;						\
		(kvar) = kh_key(h,__i);								\
		(vvar) = kh_val(h,__i);								\
		code;												\
	} }

/*! @function
  @abstract     Iterate over the values in the hash table
  @param  h     Pointer to the hash table [khash_t(name)*]
  @param  vvar  Variable to which value will be assigned
  @param  code  Block of code to execute
 */
#define kh_foreach_value(h, vvar, code) { khint_t __i;		\
	for (__i = kh_begin(h); __i != kh_end(h); ++__i) {		\
		if (!kh_exist(h,__i)) continue;						\
		(vvar) = kh_val(h,__i);								\
		code;												\
	} }

/* More conenient interfaces */

/*! @function
  @abstract     Instantiate a hash set containing integer keys
  @param  name  Name of the hash table [symbol]
 */
#define KHASH_SET_INIT_INT(name)										\
	KHASH_INIT(name, khint32_t, char, 0, kh_int_hash_func, kh_int_hash_equal)

/*! @function
  @abstract     Instantiate a hash map containing integer keys
  @param  name  Name of the hash table [symbol]
  @param  khval_t  Type of values [type]
 */
#define KHASH_MAP_INIT_INT(name, khval_t)								\
	KHASH_INIT(name, khint32_t, khval_t, 1, kh_int_hash_func, kh_int_hash_equal)

/*! @function
  @abstract     Instantiate a hash map containing 64-bit integer keys
  @param  name  Name of the hash table [symbol]
 */
#define KHASH_SET_INIT_INT64(name)										\
	KHASH_INIT(name, khint64_t, char, 0, kh_int64_hash_func, kh_int64_hash_equal)

/*! @function
  @abstract     Instantiate a hash map containing 64-bit integer keys
  @param  name  Name of the hash table [symbol]
  @param  khval_t  Type of values [type]
 */
#define KHASH_MAP_INIT_INT64(name, khval_t)								\
	KHASH_INIT(name, khint64_t, khval_t, 1, kh_int64_hash_func, kh_int64_hash_equal)

typedef const char *kh_cstr_t;
/*! @function
  @abstract     Instantiate a hash map containing const char* keys
  @param  name  Name of the hash table [symbol]
 */
#define KHASH_SET_INIT_STR(name)										\
	KHASH_INIT(name, kh_cstr_t, char, 0, kh_str_hash_func, kh_str_hash_equal)

/*! @function
  @abstract     Instantiate a hash map containing const char* keys
  @param  name  Name of the hash table [symbol]
  @param  khval_t  Type of values [type]
 */
#define KHASH_MAP_INIT_STR(name, khval_t)								\
	KHASH_INIT(name, kh_cstr_t, khval_t, 1, kh_str_hash_func, kh_str_hash_equal)

#endif /* __AC_KHASH_H */

#include <petsc/private/petscimpl.h>
/* HASHI */
#if PETSC_USE_64BIT_INDICES
KHASH_MAP_INIT_INT64(HASHI,PetscInt)
#else
KHASH_MAP_INIT_INT(HASHI,PetscInt)
#endif

typedef khash_t(HASHI) *PetscHashI;

typedef khiter_t PetscHashIIter;

#define PetscHashICreate(ht) ((ht) = kh_init(HASHI))

#define PetscHashIClear(ht)    (kh_clear(HASHI,(ht)))

#define PetscHashIDestroy(ht) if ((ht)) {kh_destroy(HASHI,(ht));(ht)=0;}

#define PetscHashIResize(ht,n) (kh_resize(HASHI,(ht),(n)))

#define PetscHashISize(ht,n)     ((n)=kh_size((ht)))

#define PetscHashIIterNext(ht,hi)  do {++(hi);} while (!kh_exist((ht),(hi)) && (hi) != kh_end((ht)))

#define PetscHashIIterBegin(ht,hi) {(hi) = kh_begin((ht));if (!kh_exist((ht),(hi))) {PetscHashIIterNext((ht),(hi));}}


#define PetscHashIIterAtEnd(ht,hi) ((hi) == kh_end((ht)))

#define PetscHashIIterGetKeyVal(ht,hi,i,ii) if (kh_exist((ht),(hi)))((i) = kh_key((ht),(hi)),(ii) = kh_val((ht),(hi))); else ((i) = -1, (ii) = -1);
#define PetscHashIIterGetKey(ht,hi,i) if (kh_exist((ht),(hi)))((i) = kh_key((ht),(hi))); else ((i) = -1);
#define PetscHashIIterGetVal(ht,hi,ii) if (kh_exist((ht),(hi)))((ii) = kh_val((ht),(hi))); else ((ii) = -1);

#define PetscHashIDel(ht,i) (kh_del(HASHI,(ht),(i)))

#define PetscHashIPut(ht,i,r,ii)                                        \
{                                                                       \
 int _3_hi;                                                             \
 ((ii)=kh_put(HASHI,(ht),(i),&_3_hi));                                  \
 (r)=_3_hi;                                                             \
}

#define PetscHashIAdd(ht,i,ii)                                          \
{                                                                       \
 khiter_t _11_hi;                                                       \
 int  _11_hr;                                                           \
 _11_hi = kh_put(HASHI,(ht),(i),&_11_hr);                               \
 kh_val((ht),_11_hi) = (ii);                                            \
}

PETSC_STATIC_INLINE PetscErrorCode PetscHashIDelKey(PetscHashI ht, PetscInt key)
{
  khiter_t hi = kh_get(HASHI, ht, key);
  if (hi != kh_end(ht)) kh_del(HASHI, ht, hi);
  return 0;
}

/*
  arr is the integer array to put the indices to, n is the offset into arr to start putting the indices at.
  n is updated as the indices are put into arr, so n must be an lvalue.
 */
PETSC_STATIC_INLINE PetscErrorCode PetscHashIGetKeys(PetscHashI ht, PetscInt *n, PetscInt arr[])
{
  PetscHashIIter hi;
  PetscInt       off = *n;

  for (hi = kh_begin(ht); hi != kh_end(ht); ++hi) {
    if (kh_exist(ht, hi) && !__ac_isdel(ht->flags, hi)) arr[off++] = kh_key(ht, hi);
  }
  *n = off;
  return 0;
}

#define PetscHashIGetVals(ht,n,arr)                                     \
{                                                                       \
  PetscHashIIter _12_hi;                                                \
  PetscInt _12_ii;                                                      \
  PetscHashIBegin((ht),_12_hi);                                         \
  while (!PetscHashIIterAtEnd((ht),_12_hi)) {                            \
    PetscHashIIterGetVal((ht),_12_hi,_12_ii);                           \
    (arr)[(n)++] = _12_ii;                                              \
    PetscHashIIterNext((ht),_12_hi);                                    \
  }                                                                     \
}

#define PetscHashIDuplicate(ht,hd)                                      \
{                                                                       \
  PetscHashIIter  _14_hi;                                               \
  PetscInt   _14_i, _14_ii;                                             \
  PetscHashICreate((hd));                                               \
  PetscHashIIterBegin((ht),_14_hi);                                     \
  while (!PetscHashIIterAtEnd((ht),_14_hi)) {                            \
    PetscHashIIterGetKeyVal((ht),_14_hi,_14_i,_14_ii);                  \
    PetscHashIAdd((hd), _14_i,_14_ii);                                  \
    PetscHashIIterNext((ht),_14_hi);                                    \
  }                                                                     \
}

#define PetscHashIHasKey(ht,i,has)                \
{                                                 \
  khiter_t _9_hi;                                 \
  _9_hi = kh_get(HASHI,(ht),(i));                 \
  if (_9_hi != kh_end((ht))) (has) = PETSC_TRUE;  \
  else                       (has) = PETSC_FALSE; \
}                                                 \

/*
 Locate index i in the hash table ht. If i is found in table, ii is its index,
 between 0 and kh_size(ht)-1 (inclusive); otherwise, ii == -1.
 */
#define PetscHashIMap(ht,i,ii)             \
{                                          \
  khiter_t _9_hi;                          \
  _9_hi = kh_get(HASHI,(ht),(i));          \
  if (_9_hi != kh_end((ht))) {             \
    (ii) = kh_val((ht),_9_hi);             \
  } else (ii) = -1;                        \
}                                          \

/*
 Locate all integers from array iarr of length len in hash table ht.
 Their images -- their local numbering -- are stored in iiarr of length len.
 If drop == PETSC_TRUE:
  - if an integer is not found in table, it is omitted and upon completion
    iilen has the number of located indices; iilen <= ilen in this case.
 If drop == PETSC_FALSE:
  - if an integer is not found in table, it is replaced by -1; iilen == ilen
    upon completion.
 */
#define PetscHashIMapArray(ht,ilen,iarr,iilen,iiarr)                   \
  do {                                                                 \
    PetscInt _10_i;                                                    \
    (iilen) = 0;                                                       \
    for (_10_i = 0, (iilen) = 0; _10_i < (ilen); ++_10_i) {            \
      PetscHashIMap(ht,(iarr)[_10_i],(iiarr)[(iilen)]);                \
      if ((iiarr)[(iilen)] != -1) ++(iilen);                           \
    }                                                                  \
} while (0)

/* HASHIJ */
/* Linked list of values in a bucket. */
struct _IJNode {
  PetscInt       k;
  struct _IJNode *next;
};
typedef struct _IJNode IJNode;

/* Value (holds a linked list of nodes) in the bucket. */
struct _IJVal {
  PetscInt n;
  IJNode   *head, *tail;
};
typedef struct _IJVal IJVal;

/* Key (a pair of integers). */
struct _PetscHashIJKey {
  PetscInt i, j;
};
typedef struct _PetscHashIJKey PetscHashIJKey;

/* Hash function: mix two integers into one.
   Shift by half the number of bits in PetscInt to the left and then XOR.  If the indices fit into the lowest half part of PetscInt, this is a bijection.
   We should shift by (8/2)*sizeof(PetscInt): sizeof(PetscInt) is the number of bytes in PetscInt, with 8 bits per byte.
 */
#define IJKeyHash(key) ((((key).i) << (4*sizeof(PetscInt)))^((key).j))

/* Compare two keys (integer pairs). */
#define IJKeyEqual(k1,k2) (((k1).i==(k2).i) ? ((k1).j==(k2).j) : 0)

KHASH_INIT(HASHIJ,PetscHashIJKey,IJVal,1,IJKeyHash,IJKeyEqual)

struct _PetscHashIJ {
  PetscBool multivalued;
  PetscInt  size;
  khash_t(HASHIJ) *ht;
};


typedef struct _PetscHashIJ *PetscHashIJ;

typedef khiter_t             PetscHashIJIter;

typedef IJNode              *PetscHashIJValIter;

PETSC_STATIC_INLINE PetscErrorCode PetscHashIJCreate(PetscHashIJ *h)
{
  PetscErrorCode _15_ierr;

  PetscFunctionBegin;
  PetscValidPointer(h,1);
  _15_ierr          = PetscNew((h));CHKERRQ(_15_ierr);
  (*h)->ht          = kh_init(HASHIJ);
  (*h)->multivalued = PETSC_TRUE;
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscHashIJGetMultivalued(PetscHashIJ h, PetscBool *m)
{
  PetscFunctionBegin;
  *m = (h)->multivalued;
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscHashIJSetMultivalued(PetscHashIJ h, PetscBool m)
{
  PetscFunctionBegin;
  (h)->multivalued = m;
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscHashIJResize(PetscHashIJ h, PetscInt n)
{
  PetscFunctionBegin;
  (kh_resize(HASHIJ,(h)->ht,(n)));
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscHashIJKeySize(PetscHashIJ h, PetscInt *n)
{
  PetscFunctionBegin;
  ((*n)=kh_size((h)->ht));
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscHashIJSize(PetscHashIJ h, PetscInt *m)
{
  PetscFunctionBegin;
  (*m)=h->size;
  PetscFunctionReturn(0);
}

/*
 Locate key i in the hash table h. If i is found in table, ii is its first value; otherwise, ii == -1.
*/
PETSC_STATIC_INLINE PetscErrorCode PetscHashIJGet(PetscHashIJ h, PetscHashIJKey i, PetscInt *ii)
{
  khiter_t _9_hi;

  PetscFunctionBegin;
  _9_hi = kh_get(HASHIJ, (h)->ht, (i));
  if (_9_hi != kh_end((h)->ht)) *ii = kh_val((h)->ht, _9_hi).head->k;
  else *ii = -1;
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscHashIJIterNext(PetscHashIJ h, PetscHashIJIter hi, PetscHashIJIter *hn)
{
  PetscFunctionBegin;
  *hn = hi;
  do { ++(*hn); } while (!kh_exist((h)->ht,(*hn)) && (*hn) != kh_end((h)->ht));
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscHashIJIterBegin(PetscHashIJ h, PetscHashIJIter *hi)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  (*hi) = kh_begin((h)->ht);if (*hi != kh_end((h)->ht) && !kh_exist((h)->ht,(*hi))) {ierr = PetscHashIJIterNext((h),(*hi),(hi));CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#define PetscHashIJIterAtEnd(h,hi) ((hi) == kh_end((h)->ht))

PETSC_STATIC_INLINE PetscErrorCode PetscHashIJIterGetKey(PetscHashIJ h, PetscHashIJIter hi, PetscHashIJKey *key)
{
  PetscFunctionBegin;
  (*key) = kh_key((h)->ht,(hi));
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscHashIJIterGetValIter(PetscHashIJ h, PetscHashIJIter hi, PetscHashIJValIter *vi)
{
  PetscFunctionBegin;
  if (hi != kh_end(h->ht) && kh_exist((h)->ht,(hi)))((*vi) = kh_val((h)->ht,(hi)).head);
  else ((*vi) = 0);
  PetscFunctionReturn(0);
}

#define PetscHashIJValIterAtEnd(h, vi) ((vi) == 0)

PETSC_STATIC_INLINE PetscErrorCode PetscHashIJValIterNext(PetscHashIJ h, PetscHashIJValIter vi, PetscHashIJValIter *vn)
{
  PetscFunctionBegin;
  ((*vn) = (vi)->next);
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscHashIJValIterGetVal(PetscHashIJ h, PetscHashIJValIter vi, PetscInt *v)
{
  PetscFunctionBegin;
  ((*v) = (vi)->k);
  PetscFunctionReturn(0);
}


PETSC_STATIC_INLINE PetscErrorCode PetscHashIJAdd(PetscHashIJ h,PetscHashIJKey i, PetscInt ii)
{
  khiter_t       _11_hi;
  int            _11_r;
  IJNode         *_11_ijnode;
  IJVal          *_11_ijval;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  _11_hi    = kh_put(HASHIJ,(h)->ht,(i),&_11_r);
  _11_ijval = &(kh_val((h)->ht,_11_hi));
  if (_11_r) {
    _11_ijval->head = _11_ijval->tail = 0;
    _11_ijval->n    = 0;
  }
  if (!_11_r && !(h)->multivalued) _11_ijval->head->k = (ii);
  else {
    ierr          = PetscNew(&_11_ijnode);CHKERRQ(ierr);
    _11_ijnode->k = (ii);
    _11_ijval     = &(kh_val((h)->ht,_11_hi));
    if (!_11_ijval->tail) {
      _11_ijval->tail = _11_ijnode;
      _11_ijval->head = _11_ijnode;
    } else {
      _11_ijval->tail->next = _11_ijnode;
      _11_ijval->tail       = _11_ijnode;
    }
    ++(_11_ijval->n);
    ++((h)->size);
  }
  PetscFunctionReturn(0);
}

/*
  arr is the key array to put the key to, and must be big enough to accommodate all keys.
 */
PETSC_STATIC_INLINE PetscErrorCode PetscHashIJGetKeys(PetscHashIJ h,PetscHashIJKey *arr)
{
  PetscHashIJIter _12_hi;
  PetscHashIJKey  _12_key;
  PetscInt        n;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  n    = 0;
  ierr = PetscHashIJIterBegin((h),&_12_hi);CHKERRQ(ierr);
  while (!PetscHashIJIterAtEnd((h),_12_hi)) {
    ierr         = PetscHashIJIterGetKey((h),_12_hi,&_12_key);CHKERRQ(ierr);
    (arr)[(n)++] = _12_key;
    ierr         = PetscHashIJIterNext((h),_12_hi, &_12_hi);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*
  iarr,jarr,karr are integer arrays to put the indices into, and must be allocated to the right size.
 */
PETSC_STATIC_INLINE PetscErrorCode PetscHashIJGetIndices(PetscHashIJ h, PetscInt *iarr, PetscInt *jarr, PetscInt *karr)
{
  PetscErrorCode     ierr;
  PetscHashIJIter    _12_hi;
  PetscHashIJValIter _12_vi;
  PetscHashIJKey     _12_key;
  PetscInt           n = 0;

  PetscFunctionBegin;
  ierr = PetscHashIJIterBegin((h),&_12_hi);CHKERRQ(ierr);
  while (!PetscHashIJIterAtEnd((h),_12_hi)) {
    ierr = PetscHashIJIterGetKey((h),_12_hi,&_12_key);CHKERRQ(ierr);
    ierr = PetscHashIJIterGetValIter((h),_12_hi,&_12_vi);CHKERRQ(ierr);
    while (!PetscHashIJValIterAtEnd((h),_12_vi)) {
      (iarr)[(n)] = _12_key.i;
      (jarr)[(n)] = _12_key.j;
      ierr        = PetscHashIJValIterGetVal((h),_12_vi,&(karr)[(n)]);CHKERRQ(ierr);
      ++(n);
      ierr = PetscHashIJValIterNext((h),_12_vi, &_12_vi);CHKERRQ(ierr);
    }
    ierr = PetscHashIJIterNext((h),_12_hi, &_12_hi);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscHashIJDuplicate(PetscHashIJ h, PetscHashIJ *hd)
{
  PetscHashIJIter    _14_hi;
  PetscHashIJValIter _14_vi;
  PetscHashIJKey     _14_key;
  PetscInt           _14_val;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscHashIJCreate((hd));CHKERRQ(ierr);
  ierr = PetscHashIJIterBegin((h),&_14_hi);CHKERRQ(ierr);
  while (!PetscHashIJIterAtEnd((h),_14_hi)) {
    ierr = PetscHashIJIterGetKey((h),_14_hi,&_14_key);CHKERRQ(ierr);
    ierr = PetscHashIJIterGetValIter((h),_14_hi,&_14_vi);CHKERRQ(ierr);
    while (!PetscHashIJValIterAtEnd((h),_14_vi)) {
      ierr = PetscHashIJValIterNext((h),_14_vi,&_14_vi);CHKERRQ(ierr);
      ierr = PetscHashIJValIterGetVal((h),_14_vi,&_14_val);CHKERRQ(ierr);
      ierr = PetscHashIJAdd((*hd), _14_key,_14_val);CHKERRQ(ierr);
      ierr = PetscHashIJValIterNext((h),_14_vi,&_14_vi);CHKERRQ(ierr);
    }
    ierr = PetscHashIJIterNext((h),_14_hi, &_14_hi);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscHashIJClearValues(PetscHashIJ h)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if ((h) && (h)->ht) {
    PetscHashIJIter    _15_hi;
    PetscHashIJValIter _15_vi, _15_vid;
    PetscErrorCode     _15_ierr;
    ierr = PetscHashIJIterBegin((h),&_15_hi);CHKERRQ(ierr);
    while (!PetscHashIJIterAtEnd((h),_15_hi)) {
      ierr = PetscHashIJIterGetValIter((h),_15_hi,&_15_vi);CHKERRQ(ierr);
      while (!PetscHashIJValIterAtEnd((h),_15_vi)) {
        _15_vid       = _15_vi;
        ierr          = PetscHashIJValIterNext((h),_15_vi,&_15_vi);CHKERRQ(ierr);
        _15_vid->next = 0;
        _15_ierr      = PetscFree(_15_vid);CHKERRQ(_15_ierr);
      }
      ierr = PetscHashIJIterNext((h),_15_hi,&_15_hi);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscHashIJClear(PetscHashIJ h)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscHashIJClearValues((h));CHKERRQ(ierr);
  kh_clear(HASHIJ,(h)->ht);
  (h)->size = 0;
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscHashIJDestroy(PetscHashIJ *h)
{
  PetscFunctionBegin;
  PetscValidPointer(h,1);
  if ((*h)) {
    PetscErrorCode _16_ierr;
    PetscHashIJClearValues((*h));
    if ((*h)->ht) {
      kh_destroy(HASHIJ,(*h)->ht);
      (*h)->ht=0;
    }
    _16_ierr = PetscFree((*h));CHKERRQ(_16_ierr);
  }
  PetscFunctionReturn(0);
}

/* HASHJK */
/* Linked list of values in a bucket. */
struct _JKNode {
  PetscInt        k;
  struct _JKNode *next;
};
typedef struct _JKNode JKNode;

/* Value (holds a linked list of nodes) in the bucket. */
struct _JKVal {
  PetscInt n;
  JKNode  *head, *tail;
};
typedef struct _JKVal JKVal;

/* Key (a quartet of integers). */
struct _PetscHashJKKey {
  PetscInt j, k;
};
typedef struct _PetscHashJKKey PetscHashJKKey;

/* Hash function: mix two integers into one.
   Shift by a quarter the number of bits in PetscInt to the left and then XOR.  If the indices fit into the lowest quarter part of PetscInt, this is a bijection.
   We should shift by (8/4)*sizeof(PetscInt): sizeof(PetscInt) is the number of bytes in PetscInt, with 8 bits per byte.
 */
#define JKKeyHash(key) ( (((key).j) << (4*sizeof(PetscInt)))^((key).k) )

/* Compare two keys (integer pairs). */
#define JKKeyEqual(k1,k2) (((k1).j==(k2).j) ? ((k1).k==(k2).k) : 0)

KHASH_INIT(HASHJK,PetscHashJKKey,JKVal,1,JKKeyHash,JKKeyEqual)

struct _PetscHashJK {
  khash_t(HASHJK) *ht;
};

typedef struct _PetscHashJK *PetscHashJK;

typedef khiter_t             PetscHashJKIter;

typedef JKNode              *PetscHashJKValIter;

PETSC_STATIC_INLINE PetscErrorCode PetscHashJKCreate(PetscHashJK *h)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(h, 1);
  ierr = PetscNew((h));CHKERRQ(ierr);
  (*h)->ht = kh_init(HASHJK);
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscHashJKResize(PetscHashJK h, PetscInt n)
{
  PetscFunctionBegin;
  (kh_resize(HASHJK, (h)->ht, (n)));
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscHashJKKeySize(PetscHashJK h, PetscInt *n)
{
  PetscFunctionBegin;
  ((*n) = kh_size((h)->ht));
  PetscFunctionReturn(0);
}

/*
  PetscHashJKPut - Insert key in the hash table

  Input Parameters:
+ h - The hash table
- key - The key to insert

  Output Parameter:
+ missing - 0 if the key is present in the hash table, 1 if the bucket is empty (never used), 2 if the element in the bucket has been deleted
- iter - Iterator into table

  Level: developer

.seealso: PetscHashJKCreate(), PetscHashJKSet()
*/
PETSC_STATIC_INLINE PetscErrorCode PetscHashJKPut(PetscHashJK h, PetscHashJKKey key, int *missing, PetscHashJKIter *iter)
{
  PetscFunctionBeginHot;
  *iter = kh_put(HASHJK, (h)->ht, (key), missing);
  PetscFunctionReturn(0);
}

/*
  PetscHashJKSet - Set the value for an iterator in the hash table

  Input Parameters:
+ h - The hash table
. iter - An iterator into the table
- value - The value to set

  Level: developer

.seealso: PetscHashJKCreate(), PetscHashJKPut(), PetscHashJKGet()
*/
PETSC_STATIC_INLINE PetscErrorCode PetscHashJKSet(PetscHashJK h, PetscHashJKIter iter, PetscInt value)
{
  PetscFunctionBeginHot;
  kh_val((h)->ht, iter).n = value;
  PetscFunctionReturn(0);
}

/*
  PetscHashJKGet - Get the value for an iterator in the hash table

  Input Parameters:
+ h - The hash table
. iter - An iterator into the table

  Output Parameters:
. value - The value to get

  Level: developer

.seealso: PetscHashJKCreate(), PetscHashJKPut(), PetscHashJKSet()
*/
PETSC_STATIC_INLINE PetscErrorCode PetscHashJKGet(PetscHashJK h, PetscHashJKIter iter, PetscInt *value)
{
  PetscFunctionBeginHot;
  *value = kh_val((h)->ht, iter).n;
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscHashJKClear(PetscHashJK h)
{
  PetscFunctionBegin;
  kh_clear(HASHJK, (h)->ht);
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscHashJKDestroy(PetscHashJK *h)
{
  PetscFunctionBegin;
  PetscValidPointer(h, 1);
  if ((*h)) {
    PetscErrorCode ierr;

    if ((*h)->ht) {
      kh_destroy(HASHJK, (*h)->ht);
      (*h)->ht = NULL;
    }
    ierr = PetscFree((*h));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* HASHIJKL */
/* Linked list of values in a bucket. */
struct _IJKLNode {
  PetscInt       k;
  struct _IJKLNode *next;
};
typedef struct _IJKLNode IJKLNode;

/* Value (holds a linked list of nodes) in the bucket. */
struct _IJKLVal {
  PetscInt n;
  IJKLNode   *head, *tail;
};
typedef struct _IJKLVal IJKLVal;

/* Key (a quartet of integers). */
struct _PetscHashIJKLKey {
  PetscInt i, j, k, l;
};
typedef struct _PetscHashIJKLKey PetscHashIJKLKey;

/* Hash function: mix two integers into one.
   Shift by a quarter the number of bits in PetscInt to the left and then XOR.  If the indices fit into the lowest quarter part of PetscInt, this is a bijection.
   We should shift by (8/4)*sizeof(PetscInt): sizeof(PetscInt) is the number of bytes in PetscInt, with 8 bits per byte.
 */
#define IJKLKeyHash(key) ( (((key).i) << (4*sizeof(PetscInt)))^((key).j)^(((key).k) << (2*sizeof(PetscInt)))^(((key).l) << (6*sizeof(PetscInt))) )

/* Compare two keys (integer pairs). */
#define IJKLKeyEqual(k1,k2) (((k1).i==(k2).i) ? ((k1).j==(k2).j) ? ((k1).k==(k2).k) ? ((k1).l==(k2).l) : 0 : 0 : 0)

KHASH_INIT(HASHIJKL,PetscHashIJKLKey,IJKLVal,1,IJKLKeyHash,IJKLKeyEqual)

struct _PetscHashIJKL {
  khash_t(HASHIJKL) *ht;
};

typedef struct _PetscHashIJKL *PetscHashIJKL;

typedef khiter_t               PetscHashIJKLIter;

typedef IJKLNode              *PetscHashIJKLValIter;

PETSC_STATIC_INLINE PetscErrorCode PetscHashIJKLCreate(PetscHashIJKL *h)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(h, 1);
  ierr = PetscNew((h));CHKERRQ(ierr);
  (*h)->ht = kh_init(HASHIJKL);
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscHashIJKLResize(PetscHashIJKL h, PetscInt n)
{
  PetscFunctionBegin;
  (kh_resize(HASHIJKL, (h)->ht, (n)));
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscHashIJKLKeySize(PetscHashIJKL h, PetscInt *n)
{
  PetscFunctionBegin;
  ((*n) = kh_size((h)->ht));
  PetscFunctionReturn(0);
}

/*
  PetscHashIJKLPut - Insert key in the hash table

  Input Parameters:
+ h - The hash table
- key - The key to insert

  Output Parameter:
+ missing - 0 if the key is present in the hash table, 1 if the bucket is empty (never used), 2 if the element in the bucket has been deleted
- iter - Iterator into table

  Level: developer

.seealso: PetscHashIJKLCreate(), PetscHashIJKLSet()
*/
PETSC_STATIC_INLINE PetscErrorCode PetscHashIJKLPut(PetscHashIJKL h, PetscHashIJKLKey key, int *missing, PetscHashIJKLIter *iter)
{
  PetscFunctionBeginHot;
  *iter = kh_put(HASHIJKL, (h)->ht, (key), missing);
  PetscFunctionReturn(0);
}

/*
  PetscHashIJKLSet - Set the value for an iterator in the hash table

  Input Parameters:
+ h - The hash table
. iter - An iterator into the table
- value - The value to set

  Level: developer

.seealso: PetscHashIJKLCreate(), PetscHashIJKLPut(), PetscHashIJKLGet()
*/
PETSC_STATIC_INLINE PetscErrorCode PetscHashIJKLSet(PetscHashIJKL h, PetscHashIJKLIter iter, PetscInt value)
{
  PetscFunctionBeginHot;
  kh_val((h)->ht, iter).n = value;
  PetscFunctionReturn(0);
}

/*
  PetscHashIJKLGet - Get the value for an iterator in the hash table

  Input Parameters:
+ h - The hash table
. iter - An iterator into the table

  Output Parameters:
. value - The value to get

  Level: developer

.seealso: PetscHashIJKLCreate(), PetscHashIJKLPut(), PetscHashIJKLSet()
*/
PETSC_STATIC_INLINE PetscErrorCode PetscHashIJKLGet(PetscHashIJKL h, PetscHashIJKLIter iter, PetscInt *value)
{
  PetscFunctionBeginHot;
  *value = kh_val((h)->ht, iter).n;
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscHashIJKLClear(PetscHashIJKL h)
{
  PetscFunctionBegin;
  kh_clear(HASHIJKL, (h)->ht);
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscHashIJKLDestroy(PetscHashIJKL *h)
{
  PetscFunctionBegin;
  PetscValidPointer(h, 1);
  if ((*h)) {
    PetscErrorCode ierr;

    if ((*h)->ht) {
      kh_destroy(HASHIJKL, (*h)->ht);
      (*h)->ht = NULL;
    }
    ierr = PetscFree((*h));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#endif /* _KHASH_H */

