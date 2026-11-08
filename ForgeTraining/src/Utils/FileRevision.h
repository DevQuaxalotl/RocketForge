#pragma once

#include <filesystem>
#include <sstream>
#include <stdexcept>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#else
#include <sys/stat.h>
#include <cerrno>
#endif

namespace RocketForge
{
/**
 * Cheap dependency identity for cached display data.
 * Windows change time and file identity also detect replacements which preserve size and modification time.
 * An empty result means missing; access errors must not mean deletion.
 */
inline std::string FileRevision(const std::filesystem::path& path)
{
#ifdef _WIN32
	const DWORD shareMode = FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE;
	HANDLE handle = CreateFileW(path.c_str(), FILE_READ_ATTRIBUTES, shareMode, nullptr, OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS, nullptr);
	if (handle == INVALID_HANDLE_VALUE) {
		const DWORD error = GetLastError();
		if (error == ERROR_FILE_NOT_FOUND || error == ERROR_PATH_NOT_FOUND) return {};
		throw std::filesystem::filesystem_error("Cannot inspect catalog dependency", path, std::error_code(error, std::system_category()));
	}

	FILE_BASIC_INFO basic{};
	BY_HANDLE_FILE_INFORMATION info{};
	const bool ok = GetFileInformationByHandleEx(handle, FileBasicInfo, &basic, sizeof(basic)) && GetFileInformationByHandle(handle, &info);
	const DWORD error = GetLastError();
	CloseHandle(handle);
	if (!ok)
		throw std::filesystem::filesystem_error("Cannot identify catalog dependency", path, std::error_code(error, std::system_category()));

	std::ostringstream stamp;
	stamp << info.dwVolumeSerialNumber << ':' << info.nFileIndexHigh << ':' << info.nFileIndexLow
		<< ':' << info.nFileSizeHigh << ':' << info.nFileSizeLow << ':' << basic.CreationTime.QuadPart
		<< ':' << basic.LastWriteTime.QuadPart << ':' << basic.ChangeTime.QuadPart << ':' << basic.FileAttributes;

	return stamp.str();
#else
	struct stat info{};
	if (::stat(path.c_str(), &info) != 0) {
		if (errno == ENOENT || errno == ENOTDIR) return {};
		throw std::filesystem::filesystem_error("Cannot inspect catalog dependency", path, std::error_code(errno, std::generic_category()));
	}

#ifdef __APPLE__
	const timespec modified = info.st_mtimespec, changed = info.st_ctimespec;
#else
	const timespec modified = info.st_mtim, changed = info.st_ctim;
#endif
	std::ostringstream stamp;
	stamp << info.st_dev << ':' << info.st_ino << ':' << info.st_size << ':' << info.st_mode
		<< ':' << modified.tv_sec << ':' << modified.tv_nsec << ':' << changed.tv_sec << ':' << changed.tv_nsec;

	return stamp.str();
#endif
}
}
