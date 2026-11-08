#pragma once
#include "MathTypes.h"

namespace RocketSim {

struct SerializeObject {
	void* ptr;
	size_t size = -1;

	template<typename T>
	SerializeObject(T& val) {
		ptr = &val;
		size = sizeof(T);
	}

	// TODO: Override for Vec/RotMat so that we don't write/read the always-0 fourth component

	SerializeObject(const SerializeObject& other) {
		ptr = other.ptr;
		size = other.size;
	}
};

// Basic struct for reading raw data from a file
struct DataStreamIn {
	std::vector<byte> data;
	size_t pos = 0;

	DataStreamIn() = default;

	DataStreamIn(std::filesystem::path filePath, bool versionCheck) {
		std::ifstream fileStream = std::ifstream(filePath, std::ios::binary);
		if (!fileStream.good())
			RS_ERR_CLOSE("Failed to read file " << filePath << ", cannot open file.");
		
		fileStream >> std::noskipws;
		data = std::vector<byte>(std::istreambuf_iterator<char>(fileStream), std::istreambuf_iterator<char>());

		if (versionCheck && !DoVersionCheck()) {
			RS_ERR_CLOSE("Failed to read file " << filePath << ", file is invalid or from a different version of RocketSim.");
		}
	}

	bool DoVersionCheck() {
		uint32_t versionID = Read<uint32_t>();
		return versionID == RS_VERSION_ID;
	}

	bool IsDone() const {
		return pos >= data.size();
	}

	bool IsOverflown() const {
		return pos > data.size();
	}

	size_t GetNumBytesLeft() const {
		if (IsDone()) {
			return 0;
		} else {
			return data.size() - pos;
		}
	}

	void ReadBytes(void* out, size_t amount) {
		if (GetNumBytesLeft() >= amount) {
			byte* asBytes = (byte*)out;
			memcpy(asBytes, data.data() + pos, amount);

			if (RS_IS_BIG_ENDIAN)
				std::reverse(asBytes, asBytes + sizeof(amount));
		}

		pos += amount;
	}

	template <typename T>
	T Read() {
		byte bytes[sizeof(T)];
		ReadBytes(bytes, sizeof(T));
		return *(T*)bytes;
	}

	template <typename T>
	void Read(T& out) {
		out = Read<T>();
	}

	void ReadMultipleFromList(std::vector<SerializeObject> objs) {
		uint32_t amount = Read<uint32_t>();
		if (amount != objs.size())
			RS_ERR_CLOSE("DataStreamIn::ReadMultipleFromList(): Prop count mismatch, expected " << objs.size() << " but have " << amount << ".");
		
		for (const SerializeObject& obj : objs)
			ReadBytes(obj.ptr, obj.size);
	}

	template <typename... Args>
	void ReadMultiple(Args&... args) {
		ReadMultipleFromList({ args... });
	}
};

// Basic struct for writing raw data to a file
struct DataStreamOut {
	std::vector<byte> data;
	size_t pos = 0;

	DataStreamOut() = default;

	void WriteBytes(const void* ptr, size_t amount) {
		data.reserve(amount);
		if (RS_IS_BIG_ENDIAN) {
			byte* reversed = (byte*)malloc(amount);
			memcpy(reversed, ptr, amount);
			std::reverse(reversed, reversed + amount);
			data.insert(data.end(), (byte*)reversed, (byte*)reversed + amount);
			free(reversed);
		} else {
			data.insert(data.end(), (byte*)ptr, (byte*)ptr + amount);
		}

		pos += amount;
	}

	template <typename T>
	void Write(const T& val) {
		WriteBytes(&val, sizeof(T));
	}

	void WriteMultipleFromList(std::vector<SerializeObject> objs) {
		Write<uint32_t>(objs.size());
		for (const SerializeObject& obj : objs)
			WriteBytes(obj.ptr, obj.size);
	}

	template<typename... Args>
	void WriteMultiple(Args... args) {
		WriteMultipleFromList({ args... });
	}

	void WriteToFile(std::filesystem::path filePath, bool writeVersionCheck) {
		std::ofstream fileStream = std::ofstream(filePath, std::ios::binary);
		if (!fileStream.good())
			RS_ERR_CLOSE("Failed to write to file " << filePath << ", cannot open file.");

		if (writeVersionCheck) {
			uint32_t version = RS_VERSION_ID;
			byte* versionBytes = (byte*)&version;
			data.insert(data.begin(), versionBytes, versionBytes + sizeof(version));
		}

		if (!data.empty())
			fileStream.write((char*)data.data(), data.size());
	}
};

template <>
inline void DataStreamOut::Write(const Vec& val) {
	WriteMultiple(val.x, val.y, val.z);
}

template <>
inline void DataStreamOut::Write(const RotMat& val) {
	WriteMultiple(val.forward, val.right, val.up);
}

}
