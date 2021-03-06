#pragma once

/* Type utilities and convenience expressions.  */

#include <cstddef>
#include <iterator>
#include <type_traits>

namespace nest {
namespace mc {
namespace util {

// Until C++14 ...

template <typename T>
using result_of_t = typename std::result_of<T>::type;

template <bool V, typename R = void>
using enable_if_t = typename std::enable_if<V, R>::type;

template <class...>
using void_t = void;

template <typename T>
using decay_t = typename std::decay<T>::type;

template <typename X>
std::size_t size(const X& x) { return x.size(); }

template <typename X, std::size_t N>
constexpr std::size_t size(X (&)[N]) { return N; }

template <typename T>
constexpr auto cbegin(const T& c) -> decltype(std::begin(c)) {
    return std::begin(c);
}

template <typename T>
constexpr auto cend(const T& c) -> decltype(std::end(c)) {
    return std::end(c);
}

template <typename T>
constexpr bool empty(const T& c) {
    return c.empty();
}

template <typename T, std::size_t N>
constexpr bool empty(const T (& c)[N]) noexcept {
    return false; // N cannot be zero
}

// Types associated with a container or sequence

template <typename Seq>
struct sequence_traits {
    using iterator = decltype(std::begin(std::declval<Seq&>()));
    using const_iterator = decltype(cbegin(std::declval<Seq&>()));
    using value_type = typename std::iterator_traits<iterator>::value_type;
    using reference = typename std::iterator_traits<iterator>::reference;
    using difference_type = typename std::iterator_traits<iterator>::difference_type;
    using size_type = decltype(size(std::declval<Seq&>()));
    // for use with heterogeneous ranges
    using sentinel = decltype(std::end(std::declval<Seq&>()));
    using const_sentinel = decltype(cend(std::declval<Seq&>()));
};

// Convenience short cuts for `enable_if`

template <typename T>
using enable_if_copy_constructible_t =
    enable_if_t<std::is_copy_constructible<T>::value>;

template <typename T>
using enable_if_move_constructible_t =
    enable_if_t<std::is_move_constructible<T>::value>;

template <typename T>
using enable_if_default_constructible_t =
    enable_if_t<std::is_default_constructible<T>::value>;

template <typename... T>
using enable_if_constructible_t =
    enable_if_t<std::is_constructible<T...>::value>;

template <typename T>
using enable_if_copy_assignable_t =
    enable_if_t<std::is_copy_assignable<T>::value>;

template <typename T>
using enable_if_move_assignable_t =
    enable_if_t<std::is_move_assignable<T>::value>;

// Iterator class test
// (might not be portable before C++17)

template <typename T, typename = void>
struct is_iterator: public std::false_type {};

template <typename T>
struct is_iterator<T, void_t<typename std::iterator_traits<T>::iterator_category>>:
    public std::true_type {};

template <typename T>
using is_iterator_t = typename is_iterator<T>::type;

// Random access iterator test

template <typename T, typename = void>
struct is_random_access_iterator: public std::false_type {};

template <typename T>
struct is_random_access_iterator<T, enable_if_t<
        std::is_same<
            std::random_access_iterator_tag,
            typename std::iterator_traits<T>::iterator_category>::value
    >> : public std::true_type {};

template <typename T>
using is_random_access_iterator_t = typename is_random_access_iterator<T>::type;

// Bidirectional iterator test

template <typename T, typename = void>
struct is_bidirectional_iterator: public std::false_type {};

template <typename T>
struct is_bidirectional_iterator<T, enable_if_t<
        std::is_same<
            std::random_access_iterator_tag,
            typename std::iterator_traits<T>::iterator_category>::value
        ||
        std::is_same<
            std::bidirectional_iterator_tag,
            typename std::iterator_traits<T>::iterator_category>::value
    >> : public std::true_type {};

template <typename T>
using is_bidirectional_iterator_t = typename is_bidirectional_iterator<T>::type;

// Forward iterator test

template <typename T, typename = void>
struct is_forward_iterator: public std::false_type {};

template <typename T>
struct is_forward_iterator<T, enable_if_t<
        std::is_same<
            std::random_access_iterator_tag,
            typename std::iterator_traits<T>::iterator_category>::value
        ||
        std::is_same<
            std::bidirectional_iterator_tag,
            typename std::iterator_traits<T>::iterator_category>::value
        ||
        std::is_same<
            std::forward_iterator_tag,
            typename std::iterator_traits<T>::iterator_category>::value
    >> : public std::true_type {};

template <typename T>
using is_forward_iterator_t = typename is_forward_iterator<T>::type;


template <typename I, typename E, typename = void, typename = void>
struct common_random_access_iterator {};

template <typename I, typename E>
struct common_random_access_iterator<
    I,
    E,
    void_t<decltype(false ? std::declval<I>() : std::declval<E>())>,
    enable_if_t<
        is_random_access_iterator<
            decay_t<decltype(false ? std::declval<I>() : std::declval<E>())>
        >::value
    >
> {
    using type = decay_t<
        decltype(false ? std::declval<I>() : std::declval<E>())
    >;
};

template <typename I, typename E>
using common_random_access_iterator_t = typename common_random_access_iterator<I, E>::type;

// NB: using this version of SFINAE instead of a void default template
// parameter because of a bug in gcc 4.9.3.
template <typename I, typename E>
struct has_common_random_access_iterator {
private:
    constexpr static bool test(...) { return false; }

    template <typename U = I>
    constexpr static bool test(typename common_random_access_iterator<U, E>::type*) { return true; }

public:
    constexpr static bool value = test(nullptr);
};

template <typename S>
struct is_sequence {
private:
    constexpr static bool test(...) { return false; }

    template <typename U = S>
    constexpr static bool test(decltype(std::begin(std::declval<U>()))*) { return true; }

public:
    constexpr static bool value = test(nullptr);
};

template <typename T>
using enable_if_sequence_t =
    enable_if_t<is_sequence<T>::value>;

// No generic lambdas in C++11, so some convenience accessors for pairs that
// are type-generic

struct first_t {
    template <typename U, typename V>
    U& operator()(std::pair<U, V>& p) {
        return p.first;
    }

    template <typename U, typename V>
    const U& operator()(const std::pair<U, V>& p) const {
        return p.first;
    }
};
constexpr first_t first{};

struct second_t {
    template <typename U, typename V>
    V& operator()(std::pair<U, V>& p) {
        return p.second;
    }

    template <typename U, typename V>
    const V& operator()(const std::pair<U, V>& p) const {
        return p.second;
    }
};
constexpr second_t second{};



} // namespace util
} // namespace mc
} // namespace nest
