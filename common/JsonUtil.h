#pragma once
#include <rapidjson/reader.h>
#include <rapidjson/writer.h>
#include <rapidjson/filereadstream.h>
#include <rapidjson/error/en.h>
#include <vector>
#include <string>
#include <map>
#include <iostream>
#include <fstream>

namespace JsonUtil
{
    struct ArrayHandler : public rapidjson::BaseReaderHandler<rapidjson::UTF8<>, ArrayHandler>
    {

        std::map<std::string, std::vector<double>> annos;

        bool Null()
        {
            return true;
        }
        bool Bool(bool b)
        {
            return true;
        }
        bool Int(int i)
        {
            return true;
        }
        bool Uint(unsigned u)
        {
            return true;
        }
        bool Int64(int64_t i)
        {
            return true;
        }
        bool Uint64(uint64_t u)
        {
            return true;
        }
        bool Double(double d)
        {
            annos[key_].push_back(d);
            return true;
        }
        bool String(const char *str, rapidjson::SizeType length, bool copy)
        {
            return true;
        }
        bool StartObject()
        {
            return true;
        }
        bool Key(const char *str, rapidjson::SizeType length, bool copy)
        {
            // cout << "Key(" << str << ", " << length << ", " << boolalpha << copy << ")" << endl;
            key_ = std::string(str, length);
            return true;
        }
        bool EndObject(rapidjson::SizeType memberCount)
        {
            return true;
        }
        bool StartArray()
        {
            // cout << "StartArray()" << endl;
            return true;
        }
        bool EndArray(rapidjson::SizeType elementCount)
        {
            // cout << "EndArray(" << elementCount << ")" << endl;
            return true;
        }

    private:
        std::string key_;
    };

    inline bool load_annos(const std::string &path, std::map<std::string, std::vector<double>> &annos)
    {
        using namespace rapidjson;
        using namespace std;

        auto *file = std::fopen(path.c_str(), "r");
        char readBuffer[65536];
        rapidjson::FileReadStream is(file, readBuffer, sizeof(readBuffer));

        ArrayHandler handler;
        Reader reader;
        auto res = reader.Parse(is, handler);

        std::fclose(file);
        if (!res)
        {
            return false;
        }
        annos = handler.annos;
        return true;
    }

    inline void write_annos(std::string path, const std::map<std::string, std::vector<double>> &annos)
    {
        using namespace rapidjson;
        using namespace std;

        ofstream out(path);
        StringBuffer s;
        Writer<StringBuffer> writer(s);
        writer.StartObject();
        for (auto &kv : annos)
        {
            writer.Key(kv.first.c_str());

            writer.StartArray();
            auto &contour = kv.second;
            for (size_t i = 0; i < contour.size(); i++)
            {
                writer.Double(contour[i]);
            }
            writer.EndArray();
        }
        writer.EndObject();
        out << s.GetString();
        out.close();
    }

} // namespace JsonUtil
