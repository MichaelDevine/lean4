import Lean
open Lean

/-- Decode two module setup files and compare the resulting `ModuleSetup`
values, so a formatting change cannot be mistaken for a content change. -/
def main (args : List String) : IO UInt32 := do
  let load (p : System.FilePath) : IO Json := do
    let text ← IO.FS.readFile p
    let json ← IO.ofExcept (Json.parse text)
    let setup : ModuleSetup ← IO.ofExcept (fromJson? json)
    return toJson setup
  let a ← load (args.headD "")
  let b ← load ((args.drop 1).headD "")
  if a == b then
    IO.println "setups are equal"
    return 0
  else
    IO.eprintln "setups differ"
    return 1
