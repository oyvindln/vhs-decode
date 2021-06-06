#ifndef JSONWAX_H
#define JSONWAX_H

/* Original author: Nikolai S | https://github.com/doublejim
 *
 * You may use this file under the terms of any of these licenses:
 * GNU General Public License version 2.0       https://www.gnu.org/licenses/gpl-2.0.html
 * GNU General Public License version 3         https://www.gnu.org/licenses/gpl-3.0.html
 */

#include <QFile>
#include <QTextStream>
#include <QCoreApplication>
#include <QDir>
#include "JsonWaxParser.h"
#include "JsonWaxEditor.h"
#include "JsonWaxSerializer.h"

class JsonWax
{
private:
    JsonWaxInternals::Parser PARSER;
    JsonWaxInternals::Editor* EDITOR = nullptr;
    QString PROGRAM_PATH;
    QString FILENAME;
    JsonWaxInternals::Serializer SERIALIZER;

public:
    typedef JsonWaxInternals::StringStyle StringStyle;
    static const StringStyle Compact = JsonWaxInternals::StringStyle::Compact;
    static const StringStyle Readable = JsonWaxInternals::StringStyle::Readable;

    typedef JsonWaxInternals::Type Type;
    static const Type Array = JsonWaxInternals::Type::Array;
    static const Type Null = JsonWaxInternals::Type::Null;
    static const Type Object = JsonWaxInternals::Type::Object;
    static const Type Value = JsonWaxInternals::Type::Value;

    JsonWax()
    {
        EDITOR = new JsonWaxInternals::Editor();
        PROGRAM_PATH = qApp->applicationDirPath();
    }

    JsonWax( const QString& fileName) : JsonWax()
    {
        if (!loadFile( fileName))
            qWarning("JsonWax-loadFile warning: the file doesn't exist: \"%s\"", fileName.toStdString().c_str());
    }

    ~JsonWax()
    {
        delete EDITOR;
    }

    int append( const QVariantList& keys, const QVariant& value)
    {
        return EDITOR->append( keys, value);
    }

    void copy( const QVariantList& keysFrom, QVariantList keysTo)
    {
        EDITOR->copy( keysFrom, EDITOR, keysTo);
    }

    void copy( const QVariantList& keysFrom, JsonWax& jsonTo, const QVariantList& keysTo)
    {
        EDITOR->copy( keysFrom, jsonTo.EDITOR, keysTo);
    }

    template <class T>
    T deserializeBytes( const QVariantList& keys, const T defaultValue = T())
    {
        if (keys.isEmpty())         // Can't deserialize from root, since it's not a value.
            return defaultValue;

        JsonWaxInternals::JsonType* element = EDITOR->getPointer( keys);

        if (element == nullptr || element->hasType != Type::Value)              // Return default value if the found JsonType is not of type Value.
            return defaultValue;

        QVariant value = static_cast<JsonWaxInternals::JsonValue*>(element)->VALUE;
        return SERIALIZER.deserializeBytes<T>( value.toString().toUtf8());
    }

    template <class T>
    void deserializeBytes( T& outputHere, const QVariantList& keys)
    {
        if (keys.isEmpty())         // Can't deserialize from root, since it's not a value.
            return;

        JsonWaxInternals::JsonType* element = EDITOR->getPointer( keys);

        if (element == nullptr || element->hasType != Type::Value)
            return;

        QVariant value = static_cast<JsonWaxInternals::JsonValue*>(element)->VALUE;
        SERIALIZER.deserializeBytes<T>( value.toString().toUtf8(), outputHere);
    }

    template <class T>
    T deserializeJson( const QVariantList& keys, T defaultValue = T())
    {
        JsonWaxInternals::JsonType* element = EDITOR->getPointer( keys);

        if (element == nullptr)
            return defaultValue;

        T value;
        SERIALIZER.deserializeJson<T>( EDITOR, keys, value);
        return value;               // A clean value with the deserialized data inserted.
    }

    template <class T>
    void deserializeJson( T& outputHere, const QVariantList& keys)
    {
        JsonWaxInternals::JsonType* element = EDITOR->getPointer( keys);

        if (element == nullptr)
            return;

        SERIALIZER.deserializeJson<T>( EDITOR, keys, outputHere);
    }

    int errorCode()
    {
        return PARSER.LAST_ERROR;
    }

    QString errorMsg()
    {
        return PARSER.errorToString();
    }

    int errorPos()  // May not be 100% accurate.
    {
        return PARSER.LAST_ERROR_POS;
    }

    bool exists( const QVariantList& keys)
    {
        return EDITOR->exists( keys);
    }

    bool fromByteArray( const QByteArray& bytes)
    {
        delete EDITOR;
        bool isWellFormed = PARSER.isWellformed( bytes);
        EDITOR = PARSER.getEditorObject();
        return isWellFormed;
    }

    bool isArray( const QVariantList& keys)
    {
        return EDITOR->isArray( keys);
    }

    bool isNullValue( const QVariantList& keys)
    {
        return EDITOR->isNullValue( keys);
    }

    bool isObject( const QVariantList& keys)
    {
        return EDITOR->isObject( keys);
    }

    bool isValue( const QVariantList& keys)
    {
        return EDITOR->isValue( keys);
    }

    QVariantList keys( const QVariantList& keys)
    {
        return EDITOR->keys( keys);
    }

    bool loadFile( const QString& fileName)
    {
        FILENAME = fileName;
        QDir dir (fileName);
        QFile qfile;
        if (dir.isRelative())
            // Note: Temporary fix for Linux
            qfile.setFileName( fileName);
        else
            qfile.setFileName( fileName);

        if (!qfile.exists())
            return false;

        QTextStream in (&qfile);
#if QT_VERSION < QT_VERSION_CHECK(6, 0, 0)
        in.setCodec("UTF-8");
#endif
        /*
            TODO: Determine correct codec. UTF-8 setting invalidates some ansi characters like "æ,ø,å".
            So make sure the read file is UTF-8 encoded, then everything will work perfectly.
        */
        qfile.open(QIODevice::ReadOnly);
        return fromByteArray( in.readAll().toUtf8());
    }

    void move( const QVariantList& keysFrom, const QVariantList& keysTo)
    {
        EDITOR->move( keysFrom, EDITOR, keysTo);
    }

    void move( const QVariantList& keysFrom, JsonWax& jsonTo, const QVariantList& keysTo)
    {
        EDITOR->move( keysFrom, jsonTo.EDITOR, keysTo);
    }

    void popFirst( const QVariantList& keys, int removeTimes = 1)
    {
        EDITOR->popFirst( keys, removeTimes);
    }

    void popLast( const QVariantList& keys, int removeTimes = 1)
    {
        EDITOR->popLast( keys, removeTimes);
    }

    void prepend( const QVariantList& keys, const QVariant& value)
    {
        EDITOR->prepend( keys, value);
    }

    void remove( const QVariantList& keys)
    {
        EDITOR->remove( keys);
    }

    bool save( StringStyle style = Readable, bool convertToCodePoints = false)
    {
        if (FILENAME.isEmpty())
        {
            qWarning("JsonWax-save error: use saveAs() if you haven't loaded a .json file. This document wasn't saved.");
            return false;
        } else {
            return saveAs( FILENAME, style, convertToCodePoints, true);
        }
    }

    bool saveAs( const QString& fileName, StringStyle style = Readable, bool convertToCodePoints = false, bool overwriteAllowed = true)
    {
        QDir dir (fileName);
        QFile qfile;
        if (dir.isRelative())
            // Note: Temporary fix for Linux
            qfile.setFileName( fileName);
        else
            qfile.setFileName( fileName);

        if (qfile.exists() && !overwriteAllowed)
            return false;

        qfile.open( QIODevice::WriteOnly | QIODevice::Text);
        QByteArray bytes = EDITOR->toByteArray( {}, style, convertToCodePoints);
        qint64 bytesWritten = qfile.write( bytes);
        qfile.close();
        return (bytesWritten == bytes.size());
    }

    template <class T>
    void serializeToBytes( const QVariantList& keys, const T& object)
    {
        EDITOR->setValue( keys, SERIALIZER.serializeToBytes<T>(object));
    }

    template <class T>
    void serializeToJson( const QVariantList& keys, const T& object)
    {
        JsonWaxInternals::Editor* serializedJson = SERIALIZER.serializeToJson<T>(object);   // Serialize QObject or other data type as a JSON-document
        serializedJson->move({0}, EDITOR, keys);                                            // with the data located at the first array position.
    }                                                                                       // Move to this editor.

    void setEmptyArray( const QVariantList& keys)
    {
        EDITOR->setEmptyArray( keys);
    }

    void setEmptyObject( const QVariantList& keys)
    {
        EDITOR->setEmptyObject( keys);
    }

    void setNull( const QVariantList& keys)
    {
        EDITOR->setValue( keys, QVariant());
    }

    void setValue( const QVariantList& keys, const QVariant& value)
    {
        EDITOR->setValue( keys, value);
    }

    int size( const QVariantList& keys = {})
    {
        return EDITOR->size( keys);
    }

    QString toString( StringStyle style = Readable, bool convertToCodePoints = false, const QVariantList& keys = {})
    {
        return EDITOR->toString( style, convertToCodePoints, keys);
    }

    Type type( const QVariantList& keys)
    {
        return EDITOR->type( keys);
    }

    QVariant value( const QVariantList& keys, const QVariant& defaultValue = QVariant())
    {
        return EDITOR->value( keys, defaultValue);
    }

};

#endif // JSONWAX_H
