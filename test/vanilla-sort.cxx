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

    sort(v.begin(), v.end());

    for (ptrdiff_t i = 1; i < v.size(); ++i)
	if (v[i - 1] > v[i])
	    cerr << "bad sorting!" << endl;

    write<T>(pdst, v);
}

int main (
    const int argc,
    const char * argv [])
{
    if (argc != 4)
    {
	cerr << "usage: "
	     << argv[0]
	     << "<uint8|uint16|uint32|float> <path/to/file> <path/to/result>"
	     << endl;

	return EXIT_FAILURE;
    }

    const char * TYPE = argv[1];

    cerr << "TYPE=" << TYPE << endl;

    if (string(TYPE) == string("uint8"))
	sortfile<uint8_t>(argv[2], argv[3]);
    else if (string(TYPE) == string("uint16"))
	sortfile<uint16_t>(argv[2], argv[3]);
    else if (string(TYPE) == string("uint32"))
	sortfile<uint32_t>(argv[2], argv[3]);
    else if (string(TYPE) == string("float"))
	sortfile<float>(argv[2], argv[3]);
    else
    {
	cerr << "ERROR invalid TYPE" << endl;

	return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
