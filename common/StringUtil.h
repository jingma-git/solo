#pragma once

#include <vector>
#include <string>

#include <sstream>
#include <algorithm>
#include <functional>
#include <cctype>

using namespace std;

namespace StringUtil
{

	inline string trim(string s)
	{

		// ref: http://stackoverflow.com/a/217605

		s.erase(s.begin(), std::find_if(s.begin(), s.end(), std::not1(std::ptr_fun<int, int>(std::isspace))));
		s.erase(std::find_if(s.rbegin(), s.rend(), std::not1(std::ptr_fun<int, int>(std::isspace))).base(), s.end());
		return s;
	}

	inline void split(string line, char delim, vector<string> &tokens)
	{

		tokens.clear();
		stringstream ss(line);
		string item;
		while (getline(ss, item, delim))
			tokens.push_back(trim(item));
	}

	inline string simplify(string s)
	{

		// ref: http://stackoverflow.com/a/5561835

		s.erase(std::unique(s.begin(), s.end(), [](char l, char r) { return (l == r) && (l == ' '); }), s.end());
		return s;
	}

	inline string toLower(string s)
	{

		// ref: http://stackoverflow.com/a/313990/1056666

		string o = s;
		std::transform(s.begin(), s.end(), o.begin(), ::tolower);
		return o;
	}

	inline string toUpper(string s)
	{

		string o = s;
		std::transform(s.begin(), s.end(), o.begin(), ::toupper);
		return o;
	}
} // namespace StringUtil