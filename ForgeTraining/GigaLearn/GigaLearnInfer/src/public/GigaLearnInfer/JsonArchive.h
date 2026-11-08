#pragma once

#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace GGL {

	class JsonArchive;

	/** True when T declares a member void Serialize(JsonArchive&). */
	template <typename T>
	concept HasSerializeMember = requires(T& value, JsonArchive& ar) { value.Serialize(ar); };

	/** True when a free SerializeFields(JsonArchive&, T&) is visible via ADL (for types we cannot modify). */
	template <typename T>
	concept HasSerializeFieldsADL = requires(T& value, JsonArchive& ar) { SerializeFields(ar, value); };

	namespace JsonArchiveDetail {

		template <typename T> struct IsVector : std::false_type {};
		template <typename T> struct IsVector<std::vector<T>> : std::true_type {};

		template <typename T> struct IsOptional : std::false_type {};
		template <typename T> struct IsOptional<std::optional<T>> : std::true_type {};

		template <typename T> struct IsStringMap : std::false_type {};
		template <typename T> struct IsStringMap<std::map<std::string, T>> : std::true_type {};
	}

	/** Static enum<->string name table for EnumField. One instance per enum, next to the enum's owner. */
	template <typename E>
	struct EnumNames {
		std::vector<std::pair<E, std::string>> entries;

		const char* NameOf(E value) const {
			for (const auto& [entryValue, entryName] : entries)
				if (entryValue == value)
					return entryName.c_str();
			return nullptr;
		}

		bool ValueOf(const std::string& name, E& out) const {
			for (const auto& [entryValue, entryName] : entries) {
				if (entryName == name) {
					out = entryValue;
					return true;
				}
			}
			return false;
		}
	};

	/**
	 * Mode-aware JSON archive: one Serialize function per type declares its fields once,
	 * and the archive either writes them to JSON (Saving) or reads them back (Loading).
	 *
	 * Load semantics: a missing key leaves the member untouched (its in-class initializer
	 * is the default); an invalid value records a path-qualified error and keeps the default.
	 * Loading never throws.
	 */
	class JsonArchive {
	public:
		static JsonArchive MakeSaver(nlohmann::json& root) {
			JsonArchive ar;
			if (!root.is_object())
				root = nlohmann::json::object();
			ar._saveNode = &root;
			return ar;
		}

		static JsonArchive MakeLoader(const nlohmann::json& root) {
			JsonArchive ar;
			ar._loadNode = &root;
			return ar;
		}

		bool IsSaving() const { return _saveNode != nullptr; }
		bool IsLoading() const { return _loadNode != nullptr; }

		/** Loading only: whether the current object has the given key. Always false when saving. */
		bool Contains(const char* key) const {
			return IsLoading() && _loadNode->is_object() && _loadNode->contains(key);
		}

		/** Serialize one field. On load, a missing key leaves 'value' untouched. */
		template <typename T>
		void Field(const char* key, T& value) {
			if (IsSaving()) {
				SaveValue((*_saveNode)[key], value);
			} else if (Contains(key)) {
				LoadValue(_loadNode->at(key), value, PathTo(key));
			}
		}

		/** Same, but applies an explicit default when the key is missing on load. */
		template <typename T, typename D>
		void Field(const char* key, T& value, const D& defaultValue) {
			if (IsLoading() && !Contains(key)) {
				value = defaultValue;
				return;
			}

			Field(key, value);
		}

		/** Like Field, but records an error when the key is missing or empty on load. */
		void RequiredField(const char* key, std::string& value) {
			Field(key, value);

			if (IsLoading() && value.empty()) {
				const std::string path = PathTo(key);
				AddError(path, "required field is missing or empty");
				_state->missingRequiredFields.push_back(path);
			}
		}

		/**
		 * Enum stored as a string via a name table. Unknown strings record an error and keep the default.
		 * On save, a value with no entry in 'names' also records an error and writes "".
		 */
		template <typename E>
		void EnumField(const char* key, E& value, const EnumNames<E>& names) {
			if (IsSaving()) {
				const char* name = names.NameOf(value);
				(*_saveNode)[key] = name ? name : "";

				if (!name)
					AddError(PathTo(key), "enum value has no entry in the name table");
			} else if (Contains(key)) {
				const nlohmann::json& node = _loadNode->at(key);

				if (!node.is_string()) {
					AddError(PathTo(key), "expected a string enum value");
					return;
				}

				const std::string text = node.get<std::string>();
				E parsed;
				if (names.ValueOf(text, parsed))
					value = parsed;
				else
					AddError(PathTo(key), "unknown enum value '" + text + "'");
			}
		}

		/** Enum stored as its underlying integer (legacy format used by debug settings and agent setup). */
		template <typename E>
		void EnumIntField(const char* key, E& value) {
			if (IsSaving()) {
				(*_saveNode)[key] = static_cast<int>(value);
			} else if (Contains(key)) {
				int raw = static_cast<int>(value);
				LoadValue(_loadNode->at(key), raw, PathTo(key));
				value = static_cast<E>(raw);
			}
		}

		/** On load, tries 'key' first, then each legacy key in order. Save writes 'key' only. */
		template <typename T>
		void FieldWithLegacy(const char* key, std::initializer_list<const char*> legacyKeys, T& value) {
			if (IsSaving() || Contains(key)) {
				Field(key, value);
				return;
			}

			for (const char* legacyKey : legacyKeys) {
				if (Contains(legacyKey)) {
					LoadValue(_loadNode->at(legacyKey), value, PathTo(key));
					return;
				}
			}
		}

		/** String field where empty saves as JSON null. On load, an explicit null clears the value to ""; a missing key leaves it untouched. */
		void FieldNullIfEmpty(const char* key, std::string& value) {
			if (IsSaving()) {
				(*_saveNode)[key] = value.empty() ? nlohmann::json(nullptr) : nlohmann::json(value);
			} else if (Contains(key)) {
				if (_loadNode->at(key).is_null())
					value.clear();
				else
					LoadValue(_loadNode->at(key), value, PathTo(key));
			}
		}

		/**
		 * Opens a nested object scope under 'key' without requiring a C++ type for it
		 * (e.g. the flat "wandb" object in segment.json). Fields on the returned archive
		 * read/write inside that object.
		 */
		JsonArchive Child(const char* key) {
			JsonArchive child(_state, PathTo(key));

			if (IsSaving()) {
				nlohmann::json& node = (*_saveNode)[key];
				if (!node.is_object())
					node = nlohmann::json::object();
				child._saveNode = &node;
			} else if (Contains(key) && _loadNode->at(key).is_object()) {
				child._loadNode = &_loadNode->at(key);
			} else {
				// Loading with no such object: child archive loads nothing (all defaults kept).
				child._loadNode = &EmptyObject();
			}

			return child;
		}

		/**
		 * Escape hatch for genuinely custom JSON shapes (e.g. car_config preset-string-or-object).
		 * Saving: the mutable object being written. Loading: nullptr.
		 */
		nlohmann::json* Raw() { return _saveNode; }

		/** Loading: the object being read. Saving: nullptr. */
		const nlohmann::json* RawLoad() const { return _loadNode; }

		void AddError(const std::string& path, const std::string& message) {
			_state->errors.push_back(path + ": " + message);
		}

		const std::vector<std::string>& GetErrors() const { return _state->errors; }

		/**
		 * Paths of fields that failed RequiredField's missing/empty check on load (a subset of
		 * GetErrors(), by path rather than message text). Lets callers that must distinguish
		 * "required field missing" from other load errors do so without matching on the error
		 * message string.
		 */
		const std::vector<std::string>& GetMissingRequiredFields() const { return _state->missingRequiredFields; }

	private:
		struct SharedState {
			std::vector<std::string> errors;
			std::vector<std::string> missingRequiredFields;
		};

		JsonArchive() : _state(std::make_shared<SharedState>()) {}

		/** Constructs a child archive sharing the parent's error state, without allocating a new one. */
		explicit JsonArchive(std::shared_ptr<SharedState> state, std::string path = {})
			: _path(std::move(path)), _state(std::move(state)) {}

		static const nlohmann::json& EmptyObject() {
			static const nlohmann::json empty = nlohmann::json::object();
			return empty;
		}

		std::string PathTo(const char* key) const {
			return _path.empty() ? std::string(key) : _path + "." + key;
		}

		/**************** Save dispatch ****************/

		template <typename T>
		void SaveValue(nlohmann::json& node, const T& value) {
			if constexpr (HasSerializeMember<T>) {
				node = nlohmann::json::object();
				JsonArchive child(_state, _path);
				child._saveNode = &node;
				// Serialize takes non-const T&; save mode never mutates members.
				const_cast<T&>(value).Serialize(child);
			} else if constexpr (HasSerializeFieldsADL<T>) {
				node = nlohmann::json::object();
				JsonArchive child(_state, _path);
				child._saveNode = &node;
				// SerializeFields takes non-const T&; save mode never mutates members.
				SerializeFields(child, const_cast<T&>(value));
			} else if constexpr (JsonArchiveDetail::IsVector<T>::value) {
				node = nlohmann::json::array();
				for (const typename T::value_type& element : value) {
					nlohmann::json elementNode;
					SaveValue(elementNode, element);
					node.push_back(std::move(elementNode));
				}
			} else if constexpr (JsonArchiveDetail::IsOptional<T>::value) {
				if (value.has_value())
					SaveValue(node, value.value());
				else
					node = nullptr;
			} else if constexpr (JsonArchiveDetail::IsStringMap<T>::value) {
				node = nlohmann::json::object();
				for (const auto& [mapKey, mapValue] : value) {
					nlohmann::json valueNode;
					SaveValue(valueNode, mapValue);
					node[mapKey] = std::move(valueNode);
				}
			} else if constexpr (std::is_same_v<T, std::filesystem::path>) {
				node = value.string();
			} else {
				node = value;
			}
		}

		/**************** Load dispatch ****************/

		template <typename T>
		void LoadValue(const nlohmann::json& node, T& value, const std::string& path) {
			if constexpr (HasSerializeMember<T>) {
				JsonArchive child(_state, path);
				child._loadNode = node.is_object() ? &node : &EmptyObject();
				value.Serialize(child);
			} else if constexpr (HasSerializeFieldsADL<T>) {
				JsonArchive child(_state, path);
				child._loadNode = node.is_object() ? &node : &EmptyObject();
				SerializeFields(child, value);
			} else if constexpr (JsonArchiveDetail::IsVector<T>::value) {
				if (!node.is_array()) {
					AddError(path, "expected an array");
					return;
				}

				value.clear();
				size_t index = 0;
				for (const nlohmann::json& elementNode : node) {
					typename T::value_type element = {};
					LoadValue(elementNode, element, path + "[" + std::to_string(index) + "]");
					value.push_back(std::move(element));
					++index;
				}
			} else if constexpr (JsonArchiveDetail::IsOptional<T>::value) {
				if (node.is_null()) {
					value.reset();
				} else {
					typename T::value_type inner = {};
					LoadValue(node, inner, path);
					value = std::move(inner);
				}
			} else if constexpr (JsonArchiveDetail::IsStringMap<T>::value) {
				if (!node.is_object()) {
					AddError(path, "expected an object");
					return;
				}

				value.clear();
				for (const auto& [mapKey, valueNode] : node.items()) {
					typename T::mapped_type mapValue = {};
					LoadValue(valueNode, mapValue, path + "." + mapKey);
					value.emplace(mapKey, std::move(mapValue));
				}
			} else if constexpr (std::is_same_v<T, std::filesystem::path>) {
				if (node.is_string())
					value = std::filesystem::path(node.get<std::string>());
				else
					AddError(path, "expected a string path");
			} else {
				try {
					value = node.get<T>();
				} catch (const std::exception& e) {
					AddError(path, std::string("invalid value: ") + e.what());
				}
			}
		}

		nlohmann::json* _saveNode = nullptr;

		// NOTE: Points at EmptyObject() when a Child scope is missing on load — never null while loading.
		const nlohmann::json* _loadNode = nullptr;

		std::string _path;
		std::shared_ptr<SharedState> _state;
	};

	/** Serializes 'value' into 'out' (out becomes a JSON object). */
	template <typename T>
	void JsonSave(const T& value, nlohmann::json& out) {
		JsonArchive ar = JsonArchive::MakeSaver(out);
		const_cast<T&>(value).Serialize(ar);
	}

	/**
	 * Loads 'out' from 'in'. Returns the collected load errors (empty on clean load).
	 * When 'missingRequiredFieldsOut' is non-null, it is filled with the paths of any
	 * RequiredField that came back missing/empty (see JsonArchive::GetMissingRequiredFields) --
	 * for callers that need to react to that specific condition without string-matching errors.
	 */
	template <typename T>
	std::vector<std::string> JsonLoad(const nlohmann::json& in, T& out, std::vector<std::string>* missingRequiredFieldsOut = nullptr) {
		JsonArchive ar = JsonArchive::MakeLoader(in);
		out.Serialize(ar);

		if (missingRequiredFieldsOut)
			*missingRequiredFieldsOut = ar.GetMissingRequiredFields();

		return ar.GetErrors();
	}

	/** Indent (spaces) used for a JSON file created by JsonSaveToFile when no existing file's style can be detected. */
	inline constexpr int kDefaultJsonIndent = 2;

	/**
	 * Returns the indent width (in spaces) that an existing pretty-printed JSON file uses, found by
	 * scanning for the first indented line. Returns 'defaultIndent' when the file is absent, unreadable,
	 * minified onto a single line, or otherwise has no detectable indentation. nlohmann discards
	 * whitespace on parse, so the on-disk style can only be recovered from the raw text.
	 */
	inline int DetectJsonIndent(const std::filesystem::path& path, int defaultIndent = kDefaultJsonIndent) {
		std::ifstream fIn(path);
		if (!fIn.good())
			return defaultIndent;

		std::string line;
		while (std::getline(fIn, line)) {
			size_t spaces = 0;
			while (spaces < line.size() && line[spaces] == ' ')
				++spaces;

			// The first line whose content is indented sits one level deep (the first key of a
			// pretty-printed object), so its leading-space count is exactly one indent step.
			if (spaces > 0 && spaces < line.size())
				return static_cast<int>(spaces);
		}

		return defaultIndent;
	}

	/**
	 * Saves 'value' to a JSON file with the given indent. Returns false with 'errorOut' set on
	 * IO failure. Field-level save errors (e.g. an enum value with no entry in its name table)
	 * do not fail the write -- the file is still valid JSON and every existing caller branches
	 * on this return value to abort/retry IO, not to catch data-modeling bugs -- but they are
	 * appended into 'errorOut' (semicolon-joined) so they are no longer silently dropped.
	 */
	template <typename T>
	bool JsonSaveToFile(const T& value, const std::filesystem::path& path, int indent, std::string* errorOut = nullptr) {
		try {
			std::filesystem::create_directories(path.parent_path());

			std::ofstream fOut(path);
			if (!fOut.good()) {
				if (errorOut) *errorOut = "Failed to open file for writing: " + path.string();
				return false;
			}

			nlohmann::json j;
			JsonArchive ar = JsonArchive::MakeSaver(j);
			const_cast<T&>(value).Serialize(ar);
			fOut << j.dump(indent);

			if (!fOut.good()) {
				if (errorOut) *errorOut = "Write failed (disk full or IO error): " + path.string();
				return false;
			}

			if (errorOut && !ar.GetErrors().empty()) {
				std::string joined;
				for (const std::string& err : ar.GetErrors()) {
					if (!joined.empty())
						joined += "; ";
					joined += err;
				}
				*errorOut = joined;
			}

			return true;
		} catch (const std::exception& e) {
			if (errorOut) *errorOut = std::string("Exception while saving: ") + e.what();
			return false;
		}
	}

	/**
	 * Saves 'value' to a JSON file, matching the indent an existing file already uses so re-saving
	 * doesn't reformat it (see DetectJsonIndent); a new file is written at kDefaultJsonIndent.
	 * Otherwise identical to the explicit-indent overload.
	 */
	template <typename T>
	bool JsonSaveToFile(const T& value, const std::filesystem::path& path, std::string* errorOut = nullptr) {
		return JsonSaveToFile(value, path, DetectJsonIndent(path), errorOut);
	}

	/**
	 * Loads 'out' from a JSON file. Returns false when the file is missing, unreadable, or
	 * unparseable (with a message in 'errorsOut'). Field-level load errors are appended to
	 * 'errorsOut' but still return true — the caller decides whether they are fatal.
	 *
	 * Contrast with JsonLoadFromFileWithSchema, which folds field-level errors into its 'ok'
	 * result: schema-gated files are expected to fully validate, so a field error there is
	 * fatal, while this plain loader leaves that judgement to the caller.
	 */
	template <typename T>
	bool JsonLoadFromFile(const std::filesystem::path& path, T& out, std::vector<std::string>& errorsOut) {
		errorsOut.clear();

		if (!std::filesystem::exists(path)) {
			errorsOut.push_back("File does not exist: " + path.string());
			return false;
		}

		std::ifstream fIn(path);
		if (!fIn.good()) {
			errorsOut.push_back("Failed to open file: " + path.string());
			return false;
		}

		nlohmann::json j = nlohmann::json::parse(fIn, nullptr, false);
		if (j.is_discarded() || !j.is_object()) {
			errorsOut.push_back("Failed to parse JSON: " + path.string());
			return false;
		}

		errorsOut = JsonLoad(j, out);
		return true;
	}

	/** Result of a schema-gated file load: 'ok' with 'value' populated, or 'errors'. */
	template <typename T>
	struct JsonFileLoadResult {
		bool ok = false;
		T value;
		std::vector<std::string> errors;
	};

	/**
	 * Loads a JSON file whose root carries a schema-version key, replacing the repeated
	 * exists/open/parse/schema-gate preamble the settings loaders used to hand-write.
	 * Any error (IO, parse, wrong or malformed schema version, or field-level load error)
	 * fails the load ('ok' == false) — unlike JsonLoadFromFile, which returns true on
	 * field-level errors and lets the caller decide whether they are fatal. Schema-gated
	 * files are expected to fully validate, so here a field error is always fatal.
	 */
	template <typename T>
	JsonFileLoadResult<T> JsonLoadFromFileWithSchema(const std::filesystem::path& path, const char* schemaKey, int expectedVersion) {
		JsonFileLoadResult<T> result;

		if (!std::filesystem::exists(path)) {
			result.errors.push_back("File does not exist: " + path.string());
			return result;
		}

		std::ifstream fIn(path);
		if (!fIn.good()) {
			result.errors.push_back("Failed to open file: " + path.string());
			return result;
		}

		nlohmann::json j = nlohmann::json::parse(fIn, nullptr, false);
		if (j.is_discarded() || !j.is_object()) {
			result.errors.push_back("Failed to parse JSON: " + path.string());
			return result;
		}

		// A present-but-non-integer schema key (e.g. hand-edited to a string) must not throw:
		// only an absent key falls back to the version-0 mismatch path below.
		if (j.contains(schemaKey) && !j.at(schemaKey).is_number_integer()) {
			result.errors.push_back("Field '" + std::string(schemaKey) + "' is not an integer schema version");
			return result;
		}

		const int schema = j.value(schemaKey, 0);
		if (schema != expectedVersion) {
			result.errors.push_back(
				"Unsupported " + std::string(schemaKey) + " " + std::to_string(schema) +
				" (this binary supports " + std::to_string(expectedVersion) + ")");
			return result;
		}

		result.errors = JsonLoad(j, result.value);
		result.ok = result.errors.empty();
		return result;
	}
}
