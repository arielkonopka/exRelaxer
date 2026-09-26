// Raw binary stream I/O (native endianness) for the serialization code. The
// byte casts iostreams require live only here; callers pass values and spans.
#pragma once
#include <cstddef>
#include <istream>
#include <ostream>
#include <span>
#include <string>
#include <type_traits>

namespace exr::binary_io {

template <typename T>
    requires std::is_trivially_copyable_v<T>
void write(std::ostream& os, std::span<const T> values)
{
    const std::span<const std::byte> bytes = std::as_bytes(values);
    os.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
}

template <typename T>
    requires std::is_trivially_copyable_v<T>
void read(std::istream& is, std::span<T> values)
{
    const std::span<std::byte> bytes = std::as_writable_bytes(values);
    is.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
}

template <typename T>
    requires std::is_trivially_copyable_v<T>
void write(std::ostream& os, const T& value)
{
    write(os, std::span<const T>(&value, 1));
}

// Reads one value; on a failed stream the result is T{} and the stream's
// state tells the caller.
template <typename T>
    requires std::is_trivially_copyable_v<T>
T read(std::istream& is)
{
    T value{};
    read(is, std::span<T>(&value, 1));
    return value;
}

inline void writeChars(std::ostream& os, const std::string& text)
{
    os.write(text.data(), static_cast<std::streamsize>(text.size()));
}

inline void readChars(std::istream& is, std::string& text)
{
    is.read(text.data(), static_cast<std::streamsize>(text.size()));
}

} // namespace exr::binary_io
