// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build !cgo || !switchyard_native

package switchyard

import (
	"os"
	"runtime"
	"testing"
)

func TestMain(m *testing.M) {
	previous := runtime.GOMAXPROCS(2)
	code := m.Run()
	runtime.GOMAXPROCS(previous)
	os.Exit(code)
}
