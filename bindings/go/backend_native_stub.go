// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//go:build !cgo || !switchyard_native

package switchyard

import (
	"encoding/json"
	"errors"
)

var errNativeUnavailable = errors.New("Switchyard native backend is unavailable; build with cgo and switchyard_native")

func nativeAvailable() bool { return false }

func backendUsesNative(engineBackend) bool { return false }

func newNativeBackend(json.RawMessage) (engineBackend, error) {
	return nil, errNativeUnavailable
}
