// Runs ScanLake Store's own format checks (src/services/format.ts) on Parquet files.
// Usage: node store_check.mjs <store repo> <format.json> <kind>=<file.parquet> ...
// Prints {"<file>": [violations...]} as JSON. Needs Node >= 22.18 (TypeScript type stripping).
import fs from 'node:fs'
import path from 'node:path'
import { pathToFileURL } from 'node:url'

const [storeRepo, formatPath, ...pairs] = process.argv.slice(2)
const { parseFormat, validateParquet } = await import(
  pathToFileURL(path.join(storeRepo, 'src', 'services', 'format.ts')).href
)

const format = parseFormat(JSON.parse(fs.readFileSync(formatPath, 'utf8')))
const result = {}
for (const pair of pairs) {
  const [kind, file] = pair.split('=')
  const buf = fs.readFileSync(file)
  const data = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength)
  result[file] = validateParquet(data, format.files[kind])
}
process.stdout.write(JSON.stringify(result))
