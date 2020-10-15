#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <chrono>

#include <cstdint>
#include <cassert>

#include <unistd.h>

using namespace std;

template <typename T>
vector<T> read (
    string pathname )
{
    cerr << "reading " << pathname << endl ;

    ifstream fin(pathname.c_str(), ios::binary);

    ptrdiff_t fsz;

    {
	fin.seekg(0, ios::end);
	fsz = fin.tellg();
	fin.seekg(0, ios::beg);
    }

    std::vector<T> retval(fsz / sizeof(T));
    fin.read((char *)retval.data(), fsz);

    return retval;
}

template <typename T>
void write (
    string pathname,
    const vector<T>& in )
{
    ofstream fout(pathname.c_str(), ios::out | ios::binary);

    fout.write((char *)in.data(), sizeof(T) * in.size());
}

template <typename T>
void sortfile (
    string psrc,
    string pdst )
{
    auto&& v = read<T>(psrc);

    auto tbegin = std::chrono::high_resolution_clock::now();

    sort(v.begin(), v.end());

    auto tend = std::chrono::high_resolution_clock::now();
    auto tts = std::chrono::duration_cast<std::chrono::milliseconds>(tend - tbegin).count();

    for (ptrdiff_t i = 1; i < v.size(); ++i)
	if (v[i - 1] > v[i])
	    cerr << "bad sorting!" << endl;

    write<T>(pdst, v);

    cout << "sorted " << v.size() << " entries in " << tts << " ms. Bye." << endl;
}

int main (
    const int argc,
    const char * argv [])
{
    if (argc != 4)
    {
	cerr << "usage: "
	     << argv[0]
	     << "<int8|uint8|int16|uint16|int32|uint32|int64|uint64|float> <path/to/file> <path/to/result>"
	     << endl;

	return EXIT_FAILURE;
    }

    const char * TYPE = argv[1];

    cerr << "TYPE=" << TYPE << endl;

    if (string(TYPE) == string("int8"))
	sortfile<int8_t>(argv[2], argv[3]);
    else if (string(TYPE) == string("uint8"))
	sortfile<uint8_t>(argv[2], argv[3]);
    else if (string(TYPE) == string("int16"))
	sortfile<int16_t>(argv[2], argv[3]);
    else if (string(TYPE) == string("uint16"))
	sortfile<uint16_t>(argv[2], argv[3]);
    else if (string(TYPE) == string("int32"))
	sortfile<int32_t>(argv[2], argv[3]);
    else if (string(TYPE) == string("uint32"))
	sortfile<uint32_t>(argv[2], argv[3]);
    else if (string(TYPE) == string("int64"))
	sortfile<int64_t>(argv[2], argv[3]);
    else if (string(TYPE) == string("uint64"))
	sortfile<uint64_t>(argv[2], argv[3]);
    else if (string(TYPE) == string("float"))
	sortfile<float>(argv[2], argv[3]);
    else if (string(TYPE) == string("double"))
	sortfile<double>(argv[2], argv[3]);
    else
    {
	cerr << "ERROR invalid TYPE" << endl;

	return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
