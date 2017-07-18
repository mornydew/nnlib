#ifndef TRAITS_H
#define TRAITS_H

namespace nnlib
{

namespace detail
{

// ignore -Wunused-value for this section; it is unused on purpose for SFINAE
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-value"

/// Check whether the given type has the serialize method. Default is false.
template <typename T, typename = int>
struct HasSerialize : std::false_type
{};

/// Check whether the given type has the serialize method. This override determines it does.
template <typename T>
struct HasSerialize<T, decltype(&T::template serialize<T>, 0)> : std::true_type
{};

/// Check whether the given type has the load and save methods. Default is false.
template <typename T, typename = int, typename = int>
struct HasLoadAndSave : std::false_type
{};

/// Check whether the given type has the load and save methods. This override determines it does.
template <typename T>
struct HasLoadAndSave<T, decltype(&T::template load<T>, 0), decltype(&T::template save<T>, 0)> : std::true_type
{};

#pragma GCC diagnostic pop

}

}

#endif